
# data_io.py — I/O helpers
import io
import pandas as pd
from typing import List, Tuple, Optional

def load_input_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)

def save_long_format_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["post_text", "text", "caption", "content"]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic: choose the longest-average-length string column
    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        return None
    avg_len = {c: df[c].dropna().str.len().mean() for c in str_cols}
    return max(avg_len, key=avg_len.get) if avg_len else None
