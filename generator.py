
# generator.py — brand profile + robust generation
import os, re, json
from typing import List, Dict, Any, Tuple

try:
    from langchain.chat_models import ChatOpenAI as LCChatOpenAI_old  # type: ignore
except Exception:
    LCChatOpenAI_old = None
try:
    from langchain_openai import ChatOpenAI as LCChatOpenAI_new  # type: ignore
except Exception:
    LCChatOpenAI_new = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def extract_url_text(url: str) -> str:
    try:
        import trafilatura  # type: ignore
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            txt = trafilatura.extract(downloaded) or ""
            return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        pass
    return f"Article at {url}"

PLATFORM_LIMITS = {
    "LinkedIn": 3000,
    "X (Twitter)": 280,
    "Facebook": 63206,
    "Instagram (caption)": 2200
}

def _clip_to_limit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    if "." in trimmed[-60:]:
        pos = trimmed.rfind(".")
        if pos > 0 and pos > limit - 120:
            return trimmed[:pos+1]
    return trimmed[:-1].rstrip() + "…"

# --- backend init ---
def _normalize_content(msg) -> str:
    try:
        content = getattr(msg, "content", None)
        if content is not None:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                if texts:
                    return "\n".join(texts)
                return json.dumps(content)
        if hasattr(msg, "choices"):
            choice = msg.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content
        return str(msg)
    except Exception:
        return str(msg)

def _init_langchain_backend(api_key: str, model_name: str, base_url: str, temperature: float, max_tokens: int):
    Klass = LCChatOpenAI_new or LCChatOpenAI_old
    if not Klass:
        return None
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    kwargs = {"temperature": temperature}
    init_vars = getattr(getattr(Klass, "__init__"), "__code__", None)
    co = getattr(init_vars, "co_varnames", ())
    if "model" in co:
        kwargs["model"] = model_name
    elif "model_name" in co:
        kwargs["model_name"] = model_name
    if "max_tokens" in co:
        kwargs["max_tokens"] = max_tokens
    if base_url and "base_url" in co:
        kwargs["base_url"] = base_url
    if "api_key" in co:
        kwargs["api_key"] = api_key
    elif "openai_api_key" in co:
        kwargs["openai_api_key"] = api_key

    lc = Klass(**kwargs)
    def _call(prompt: str):
        return lc.invoke(prompt)
    return _call

def _init_openai_backend(api_key: str, model_name: str, base_url: str, temperature: float, max_tokens: int):
    if not api_key:
        raise ValueError("The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable")
    os.environ["OPENAI_API_KEY"] = api_key
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai` or enable LangChain path.")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def _call(prompt: str):
        return client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _call

def init_llm_backend(api_key: str, model_name: str, base_url: str = None, use_langchain: bool = True, temperature: float = 0.7, max_tokens: int = 220):
    if use_langchain and (LCChatOpenAI_new or LCChatOpenAI_old):
        lc = _init_langchain_backend(api_key, model_name, base_url, temperature, max_tokens)
        if lc:
            return lambda p: _normalize_content(lc(p))
    oa = _init_openai_backend(api_key, model_name, base_url, temperature, max_tokens)
    return lambda p: _normalize_content(oa(p))

# --- brand training ---
def derive_brand_profile(llm, training_texts: List[str], max_examples: int = 5) -> Dict[str, Any]:
    sample = "\n\n".join(training_texts[:100])  # cap
    prompt = (
        "You are a brand voice analyst. Given real social posts below, summarize the brand's voice.\n"
        "Return three sections in JSON with keys: style_bullets (string bullets), do_dont (string bullets), examples (3-5 short example posts)."
        "Source posts:\n"
        f"{sample}"
    )
    raw = llm(prompt)
    # Try JSON parse
    data = {"style_bullets": "", "do_dont": "", "examples": ""}
    try:
        obj = json.loads(raw)
        for k in data:
            if k in obj:
                data[k] = obj[k]
    except Exception:
        # Fallback: keep raw as style text
        data["style_bullets"] = raw
    return data

# --- prompt + generation ---
def build_prompt(
    url: str,
    platform: str,
    tone: str,
    extras: List[str],
    opening_hook: str = "",
    brand: str = "",
    custom_hashtags: str = "",
    row_metadata: Dict[str, Any] = None,
    brand_profile: Dict[str, Any] = None
) -> str:
    article_text = extract_url_text(url)
    extras_text = ", ".join(extras) if extras else "None"
    hashtags_hint = ("Use these hashtags if sensible: " + custom_hashtags) if custom_hashtags else "Use 2–5 relevant hashtags if platform-appropriate."
    hook_line = (opening_hook + "\n\n") if opening_hook else ""
    brand_line = ("Brand/Company: " + brand + "\n") if brand else ""

    notes_line = ""
    if row_metadata:
        aux = {k: v for k, v in row_metadata.items() if k not in ("URL", "postId") and str(v).strip()}
        if aux:
            notes_line = "Additional dataset context: " + str(aux) + "\n"

    limit = PLATFORM_LIMITS.get(platform, 500)
    platform_rules = {
        "LinkedIn": "Aim for 4–8 short lines. Use whitespace for readability. Light emojis ok. Add value/insight.",
        "X (Twitter)": "Keep it tight. 1–2 short sentences + strong hook. Hashtags: max 2. Emojis minimal.",
        "Facebook": "Conversational, friendly. 2–3 short paragraphs okay. One CTA.",
        "Instagram (caption)": "Compact, emotive. Break into short lines. Hashtags at the end."
    }.get(platform, "Follow platform norms.")

    brand_section = ""
    if brand_profile:
        style_bullets = brand_profile.get("style_bullets", "")
        examples = brand_profile.get("examples", "")
        do_dont = brand_profile.get("do_dont", "")
        brand_section = (
            "Brand Voice Summary (use consistently):\n"
            f"{style_bullets}\n\n"
            "Do / Don't Guidance:\n"
            f"{do_dont}\n\n"
            "Inspiration Examples:\n"
            f"{examples}\n\n"
        )

    template = (
        "You are an expert social media copywriter.\n\n"
        f"{brand_section}"
        "Goal: Write {platform}-ready post variations to promote the article below.\n"
        "Tone: {tone}\n"
        "Platform rules: {platform_rules}\n"
        "Extras to apply: {extras_text}\n"
        "{hashtags_hint}\n\n"
        "{brand_line}{notes_line}{hook_line}"
        "Article context (brief summary for reference):\n"
        "\"\"\"\n"
        "{article_excerpt}\n"
        "\"\"\"\n\n"
        "Constraints:\n"
        "- Character limit: ~{limit} characters; stay under it.\n"
        "- Avoid clickbait; be compelling, specific, and trustworthy.\n"
        "- Each variant should emphasize a different angle (benefit, stat, quote, insight, outcome, story).\n"
        "- Include a clear CTA to read the article (assume the link is attached).\n"
        "- Respect platform norms (line breaks, hashtags, emojis density).\n\n"
        "Return ONLY the variants in a numbered list, each on its own paragraph.\n"
    )

    return template.format(
        platform=platform,
        tone=tone.lower(),
        platform_rules=platform_rules,
        extras_text=extras_text,
        hashtags_hint=hashtags_hint,
        brand_line=brand_line,
        notes_line=notes_line,
        hook_line=hook_line,
        article_excerpt=article_text[:3000],
        limit=limit
    )

def _postprocess_variants(raw: str) -> List[str]:
    parts = re.split(r"\n\s*\d+[\).:-]\s*", (raw or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\n+", (raw or "").strip()) if p.strip()]
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def generate_posts_for_row(row: Dict[str, Any], prompt: str, n_variants: int, llm, platform: str, debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug_payload = {"prompt_first_800": prompt[:800]}
    batched = prompt.replace("Return ONLY the variants", f"Return ONLY **{n_variants}** variants")
    raw = llm(batched)
    debug_payload["raw_first_200"] = (raw or "")[:200]
    variants = _postprocess_variants(raw)

    if len(variants) < n_variants:
        needed = n_variants - len(variants)
        extra_raw = llm(prompt.replace("Return ONLY the variants", f"Return ONLY **{needed}** additional variants different from previous ones"))
        debug_payload["raw_extra_200"] = (extra_raw or "")[:200]
        variants.extend(_postprocess_variants(extra_raw))

    if not variants:
        variants = [f"[No content returned from LLM for URL={row.get('URL','')}. Check API key, base URL, model name, or increase max_tokens.]"]

    variants = variants[:n_variants]

    limit = PLATFORM_LIMITS.get(platform, 500)
    output = []
    for i, text in enumerate(variants, start=1):
        clipped = _clip_to_limit(text, limit)
        output.append({
            "postId": row.get("postId", ""),
            "variant": i,
            "platform": platform,
            "post_text": clipped
        })

    return (output, debug_payload if debug else {})
