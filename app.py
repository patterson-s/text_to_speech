# app.py
"""
Streamlit app: upload a .txt or .md file → receive a single MP3.
Requires openai‑python ≥ 1.16.0 (1.x SDK with TTS support).
"""

import os
import re
import html
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI
import markdown

# ───────────────────────── Configuration ─────────────────────────
MAX_CHARS = 4_000                 # TTS limit is 4 096 characters
VOICES = ["alloy", "nova", "shimmer", "fable", "echo", "onyx"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ──────────────────────── Helper utilities ───────────────────────
def strip_markdown(md: str) -> str:
    """Very lightweight Markdown → plain‑text conversion."""
    html_text = markdown.markdown(md)
    plain = re.sub(r"<[^>]+>", " ", html_text)
    return html.unescape(plain)


def read_uploaded(uploaded) -> str:
    raw = uploaded.read().decode("utf‑8", errors="replace")
    return strip_markdown(raw) if uploaded.name.lower().endswith(".md") else raw


def chunk_text(text: str, size: int = MAX_CHARS):
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n and text[end] not in {" ", "\n"}:
            end = text.rfind(" ", start, end) or end
        yield text[start:end].strip()
        start = end


def synthesize(text: str, model: str, voice: str, outfile: Path):
    """Stream each ≤4 k‑char chunk to *outfile* as MP3."""
    for i, block in enumerate(chunk_text(text)):
        file_mode = "wb" if i == 0 else "ab"
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=block,
            response_format="mp3",
        ) as resp, open(outfile, file_mode) as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)

# ────────────────────────── Streamlit UI ─────────────────────────
st.set_page_config(page_title="🗣️ Text → Speech (mp3)")
st.title("🗣️ Text → Speech (mp3)")

api_key_in = st.text_input(
    "OpenAI API key (leave blank to use your environment variable)", type="password"
)
if api_key_in:
    client.api_key = api_key_in

up_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
voice = st.selectbox("Voice", VOICES)
model = "tts-1-hd" if st.radio(
    "Model quality", ["Fast (tts-1)", "High‑quality (tts-1-hd)"],
    horizontal=True
).endswith("hd") else "tts-1"

if st.button("Generate") and up_file:
    with st.spinner("Synthesising…"):
        text = read_uploaded(up_file)
        tmp_dir = tempfile.TemporaryDirectory()
        mp3_path = Path(tmp_dir.name) / f"{Path(up_file.name).stem}.mp3"

        try:
            synthesize(text, model, voice, mp3_path)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
        else:
            data = mp3_path.read_bytes()
            st.audio(data, format="audio/mp3")
            st.download_button("Download MP3", data, mp3_path.name, "audio/mpeg")

        tmp_dir.cleanup()
