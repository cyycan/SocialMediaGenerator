
# app_nosecrets.py — Brand-trained generator (batch + single), no st.secrets usage
import os
import pandas as pd
import streamlit as st

from data_io import (
    load_input_excel,
    ensure_required_columns,
    save_long_format_csv,
    detect_text_column,
)
from generator import (
    init_llm_backend,
    build_prompt,
    generate_posts_for_row,
    derive_brand_profile,
)

st.set_page_config(page_title="Social Media Post Generator", page_icon="🧃", layout="wide")
st.title("🧃 Social Media Generator")
st.caption(" 1) Enter API keys for OPENAI (make sure to hit enter) 2) Upload a **training file** to learn your brand voice. 3) Generate posts for **new data** in batch or one‑off.")

# --- Sidebar: Model + Style Settings ---
with st.sidebar:
    st.header("Model Settings")
    api_key_input = st.text_input("OpenAI-compatible API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url_input = st.text_input("Base URL (optional, for OpenAI‑compatible endpoints)", value=os.getenv("OPENAI_BASE_URL", ""))
    model_name_input = st.text_input("Model name", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    use_langchain = st.checkbox("Use LangChain backend (ChatOpenAI)", value=True)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens per variant", 64, 1000, 300, 10)

    st.divider()
    st.header("Style Controls (applied to NEW posts)")
    platform = st.selectbox("Platform", ["LinkedIn", "X (Twitter)", "Facebook", "Instagram (caption)"])
    tone = st.selectbox("Tone", ["Professional", "Casual", "Playful", "Inspiring", "Analytical"])
    extras = st.multiselect("Extras", ["Emojis", "Hashtags", "Call-to-Action", "Short + Punchy", "Longform Teaser"], default=["Call-to-Action"])
    variants_per_url = st.slider("Variants per URL", 1, 8, 4)
    opening_hook = st.text_input("Opening hook (optional)")
    brand_name = st.text_input("Brand/Company (optional)")
    custom_hashtags = st.text_input("Custom hashtags (comma-separated, optional)")

    st.divider()
    st.header("Runtime")
    live_preview = st.checkbox("Show live preview per row", value=True)
    debug_mode = st.checkbox("Debug mode (show prompt + raw snippet)", value=False)

API_KEY = api_key_input or os.getenv("OPENAI_API_KEY")
BASE_URL = base_url_input or os.getenv("OPENAI_BASE_URL")
MODEL_NAME = model_name_input or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

# Keep brand profile in session
if "brand_profile" not in st.session_state:
    st.session_state.brand_profile = None

# --- STEP 1: Upload training data to learn brand voice ---
st.subheader("Step 1 — Upload Brand Training Data (.xlsx)")
st.write("This file should contain your **existing brand posts**. We'll infer a brand voice and pull a few style examples. (We do **not** generate for this data.)")
train_file = st.file_uploader("Choose training Excel", type=["xlsx"], key="train")
if train_file is not None:
    df_train = load_input_excel(train_file)
    text_col = detect_text_column(df_train)
    if not text_col:
        st.error("Couldn't find a text column (tried: post_text, text, caption, content). Please add one of these column names.")
    else:
        st.success(f"Training rows loaded: {len(df_train)} • Detected text column: `{text_col}`")
        with st.expander("Preview training data", expanded=False):
            st.dataframe(df_train[[text_col]].head(50), use_container_width=True)

        if st.button("🧠 Build/Update Brand Profile", type="primary"):
            try:
                llm = init_llm_backend(
                    api_key=API_KEY,
                    model_name=MODEL_NAME,
                    base_url=BASE_URL,
                    use_langchain=use_langchain,
                    temperature=0.3,  # analysis/summarization is better with lower temp
                    max_tokens=600
                )
            except Exception as e:
                st.error(f"Failed to initialize LLM backend for brand training: {e}")
                st.stop()

            st.info("Deriving brand voice…")
            st.session_state.brand_profile = derive_brand_profile(llm, df_train[text_col].dropna().astype(str).tolist(), max_examples=5)
            st.success("Brand profile created! See summary below.")

# Show current brand profile
st.subheader("Current Brand Profile")
if st.session_state.brand_profile:
    bp = st.session_state.brand_profile
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Voice & Style (bullets)**")
        st.write(bp.get("style_bullets", "—"))
        st.markdown("**Do / Don't**")
        st.write(bp.get("do_dont", "—"))
    with col2:
        st.markdown("**Examples (from training)**")
        st.write(bp.get("examples", "—"))
else:
    st.info("No brand profile yet. Upload training data and click **Build/Update Brand Profile**.")

st.divider()

# --- STEP 2: Generate for NEW data (batch or single) ---
st.subheader("Step 2 — Generate Posts for NEW Data")
tab_batch, tab_single = st.tabs(["Batch via File Upload", "Single via UI"])

# --- Batch tab ---
with tab_batch:
    st.markdown("Upload NEW content inputs (.xlsx). Required columns: **postId**, **URL**. Optional columns (e.g., *notes*, *topic*, *audience*) will be passed to the prompt.")
    new_file = st.file_uploader("Choose NEW data Excel", type=["xlsx"], key="new")
    if new_file is not None:
        df_new = load_input_excel(new_file)
        ok, missing = ensure_required_columns(df_new, required=["postId", "URL"])
        if not ok:
            st.error(f"Missing required columns: {missing}.")
        else:
            st.success(f"Loaded {len(df_new)} new rows.")
            with st.expander("Preview NEW data", expanded=False):
                st.dataframe(df_new.head(100), use_container_width=True)

            # Initialize LLM backend for generation
            try:
                llm = init_llm_backend(
                    api_key=API_KEY,
                    model_name=MODEL_NAME,
                    base_url=BASE_URL,
                    use_langchain=use_langchain,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                st.error(f"Failed to initialize LLM backend: {e}")
                st.stop()

            go = st.button("🚀 Generate for NEW Data", type="primary", key="gen_batch")
            if go:
                rows = []
                progress = st.progress(0.0)
                status = st.empty()
                live_area = st.container() if live_preview else None

                total = len(df_new)
                for i, row in df_new.iterrows():
                    prompt = build_prompt(
                        url=row.get("URL", ""),
                        platform=platform,
                        tone=tone,
                        extras=extras,
                        opening_hook=opening_hook,
                        brand=brand_name,
                        custom_hashtags=custom_hashtags,
                        row_metadata=row.to_dict(),
                        brand_profile=st.session_state.brand_profile
                    )

                    try:
                        out_rows, debug_payload = generate_posts_for_row(
                            row=row.to_dict(),
                            prompt=prompt,
                            n_variants=variants_per_url,
                            llm=llm,
                            platform=platform,
                            debug=debug_mode
                        )
                        rows.extend(out_rows)
                        if live_preview and out_rows:
                            live_area.subheader(f"Row {i+1} • postId={row.get('postId','')}")
                            live_area.dataframe(pd.DataFrame(out_rows), use_container_width=True)
                            if debug_mode and debug_payload:
                                live_area.code(str(debug_payload)[:2000], language="json")
                    except Exception as gen_err:
                        error_row = {
                            "postId": row.get("postId", ""),
                            "variant": 1,
                            "platform": platform,
                            "post_text": f"[Generation error for this row] {gen_err}"
                        }
                        rows.append(error_row)
                        if live_preview:
                            live_area.subheader(f"Row {i+1} • postId={row.get('postId','')} (error)")
                            live_area.dataframe(pd.DataFrame([error_row]), use_container_width=True)

                    progress.progress((i + 1) / total)
                    status.info(f"Processed {i + 1} / {total}")

                progress.empty()
                status.empty()

                if not rows:
                    st.warning("No outputs were produced. Enable Debug Mode and try again.")
                    st.stop()

                results_df = pd.DataFrame(rows)
                st.subheader("Results — NEW Data")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = save_long_format_csv(results_df)
                st.download_button(
                    label="⬇️ Download CSV (long format)",
                    data=csv_bytes,
                    file_name="social_posts_long_format.csv",
                    mime="text/csv"
                )
                st.toast("Done! Your CSV is ready.", icon="✅")

# --- Single tab ---
with tab_single:
    st.markdown("Enter a single URL + optional notes to generate a few variants right here.")
    single_url = st.text_input("URL to promote")
    single_notes = st.text_area("Optional notes/context (comma- or line-separated)")
    single_go = st.button("✨ Generate One-Off", type="primary", key="gen_single")

    if single_go:
        if not single_url:
            st.error("Please provide a URL.")
        else:
            try:
                llm = init_llm_backend(
                    api_key=API_KEY,
                    model_name=MODEL_NAME,
                    base_url=BASE_URL,
                    use_langchain=use_langchain,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                st.error(f"Failed to initialize LLM backend: {e}")
                st.stop()

            row_meta = {"postId": "single", "URL": single_url, "notes": single_notes}
            prompt = build_prompt(
                url=single_url,
                platform=platform,
                tone=tone,
                extras=extras,
                opening_hook=opening_hook,
                brand=brand_name,
                custom_hashtags=custom_hashtags,
                row_metadata=row_meta,
                brand_profile=st.session_state.brand_profile
            )
            out_rows, debug_payload = generate_posts_for_row(
                row=row_meta,
                prompt=prompt,
                n_variants=variants_per_url,
                llm=llm,
                platform=platform,
                debug=debug_mode
            )
            st.subheader("Single — Results")
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True)
            if debug_mode and debug_payload:
                st.subheader("Debug")
                st.code(str(debug_payload)[:2000], language="json")
