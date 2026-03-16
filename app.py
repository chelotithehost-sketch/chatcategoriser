import streamlit as st
import zipfile
import json
import io
import re
import time
import random
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Chat Categoriser · HostAfrica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
section[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.app-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b262c 100%);
    border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: #fff; }
.app-header p  { margin: 0; font-size: 0.9rem; color: #94a3b8; }
.metric-card {
    background: #1e2130; border-radius: 10px;
    padding: 1rem 1.25rem; text-align: center; border: 1px solid #2d3349;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
.provider-badge {
    display: inline-block; padding: .2rem .65rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: .04em; vertical-align: middle;
}
.badge-openai { background: #10a37f22; color: #10a37f; border: 1px solid #10a37f55; }
.badge-gemini { background: #4285f422; color: #4285f4; border: 1px solid #4285f455; }
.sec-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #1e2130; padding-bottom: .4rem; margin: 1.5rem 0 .8rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — Provider selection & keys
# ═══════════════════════════════════════════════════════════════

def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


with st.sidebar:
    st.markdown("### 🤖 AI Provider")
    provider = st.radio("Choose provider", ["OpenAI", "Gemini"], horizontal=True)

    st.markdown("---")

    if provider == "OpenAI":
        st.markdown("### 🔑 OpenAI API Key")
        _oai_secret = _get_secret("OPENAI_API_KEY")
        if _oai_secret:
            st.success("🔒 Key loaded from Streamlit Secrets")
            api_key = _oai_secret
        else:
            api_key = st.text_input(
                "Paste your key", type="password", placeholder="sk-...",
                help="Or set OPENAI_API_KEY in Streamlit Cloud Secrets."
            )
        st.markdown("### ⚙️ Model")
        model_name = st.selectbox("OpenAI model", [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ])

    else:  # Gemini
        st.markdown("### 🔑 Gemini API Key")
        _gem_secret = _get_secret("GEMINI_API_KEY")
        if _gem_secret:
            st.success("🔒 Key loaded from Streamlit Secrets")
            api_key = _gem_secret
        else:
            api_key = st.text_input(
                "Paste your key", type="password", placeholder="AIza...",
                help="Or set GEMINI_API_KEY in Streamlit Cloud Secrets."
            )
        st.markdown("### ⚙️ Model")
        model_name = st.selectbox("Gemini model", [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemma-3-27b",
            "gemini-3-flash",
        ])

    st.markdown("---")
    st.markdown("### 📊 Analysis Settings")
    sample_size = st.slider(
        "Chats to analyse (sample)", min_value=10, max_value=500, value=100, step=10,
        help="How many chats the AI reads to build the schema."
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Extracts tawk.to JSON from nested ZIPs, sends a sample to AI, "
        "and returns a clean category schema — free from procedural noise like PIN checks."
    )


# ── Header ────────────────────────────────────────────────────
badge_html = (
    '<span class="provider-badge badge-openai">OpenAI</span>'
    if provider == "OpenAI" else
    '<span class="provider-badge badge-gemini">Gemini</span>'
)
st.markdown(f"""
<div class="app-header">
  <div>
    <h1>🧠 AI Chat Categoriser &nbsp;{badge_html}</h1>
    <p>Upload your tawk.to ZIP export → get a clean, AI-derived issue taxonomy</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# AI CALL ABSTRACTION
# ═══════════════════════════════════════════════════════════════

def call_ai(prompt: str, max_tokens: int = 8192, temperature: float = 0.2) -> str:
    """Call the selected provider and return raw text."""
    if provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},  # enforces valid JSON
        )
        return response.choices[0].message.content

    else:  # Gemini
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_jsons_from_zip(uploaded_file) -> list[dict]:
    results = []

    def _process(zip_bytes: bytes, depth: int = 0):
        if depth > 10:
            return
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    data = zf.read(name)
                    if name.lower().endswith(".zip"):
                        _process(data, depth + 1)
                    elif name.lower().endswith(".json"):
                        try:
                            obj = json.loads(data.decode("utf-8", errors="replace"))
                            if isinstance(obj, list):
                                results.extend(obj)
                            elif isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
        except zipfile.BadZipFile:
            pass

    _process(uploaded_file.read())
    return results


def flatten_chat(chat: dict) -> str:
    lines = []
    if chat.get("id"):
        lines.append(f"[ID: {chat['id']}]")
    messages = chat.get("messages", chat.get("conversation", []))
    for msg in messages:
        sender = msg.get("name") or msg.get("sender") or msg.get("type", "?")
        text = msg.get("msg") or msg.get("message") or msg.get("text", "")
        if text:
            lines.append(f"{sender}: {text}")
    if len(lines) <= 1:
        lines.append(json.dumps(chat, ensure_ascii=False)[:800])
    return "\n".join(lines)


def build_schema_prompt(chat_samples: list[str]) -> str:
    sample_block = "\n\n---CHAT---\n".join(chat_samples[:200])
    return f"""
You are an expert customer support analyst at a web hosting company (HostAfrica).
You have been given a sample of live support chat transcripts.

YOUR TASK:
Analyse these chats and produce a CLEAN, DATA-DRIVEN issue categorisation schema.

STRICT RULES:
1. Categories must reflect genuine customer problems — NOT internal procedures.
   - "Support PIN" / "security PIN" is a ROUTINE authentication step at the START
     of nearly every chat. It is NOT a customer issue. Do NOT create a category for it.
   - Do not create categories for agent greetings, closing pleasantries, or any scripted steps.
2. Create 10–22 categories maximum. Merge thin or overlapping categories.
3. Each category object must contain:
   - "name": short clear title (e.g. "Email Delivery Failures")
   - "description": 1–2 sentences on what belongs here
   - "signals": list of 8–20 lowercase keyword/phrase triggers
   - "exclude": list of 3–8 phrases that look similar but must NOT trigger this category
   - "example_intents": 2–3 verbatim-style example sentences a customer might say
4. Return a top-level "meta" object containing:
   - "schema_version": "1.0"
   - "derived_from_sample_size": {len(chat_samples)}
   - "analyst_notes": paragraph explaining dominant patterns and what the old
     keyword-matching approach was getting wrong

Return ONLY a valid JSON object with keys "categories" and "meta". No preamble, no markdown fences.

=== CHAT SAMPLE ===
---CHAT---
{sample_block}
"""


def build_relabelling_prompt(schema: dict, chat_text: str) -> str:
    categories = [c["name"] for c in schema.get("categories", [])]
    cat_list = "\n".join(f"- {c}" for c in categories)
    return f"""
You are a support chat classifier. Use ONLY the categories listed below.

AVAILABLE CATEGORIES:
{cat_list}
- General (only if nothing else matches)

CHAT:
{chat_text[:1500]}

Return ONLY valid JSON:
{{"category": "<category name>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}
"""


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "🧠 Build Schema", "🏷️ Re-label Chats"])


# ──────────────────────────────────────────────────────────────
# TAB 1 — Upload
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-header">Upload your tawk.to ZIP export</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a ZIP file here (nested ZIPs handled automatically)",
        type=["zip"],
    )

    if uploaded:
        with st.spinner("🔍 Unpacking ZIPs and reading JSON files…"):
            chats = extract_jsons_from_zip(uploaded)

        if not chats:
            st.error("No JSON chat records found. Double-check the ZIP structure.")
        else:
            st.session_state["chats"] = chats

            msg_count = sum(len(c.get("messages", c.get("conversation", []))) for c in chats)
            agents = {
                m.get("name", "")
                for c in chats
                for m in c.get("messages", c.get("conversation", []))
                if m.get("type") in ("agent", "staff")
            }
            col1, col2, col3, col4 = st.columns(4)
            for col, val, lbl in [
                (col1, f"{len(chats):,}", "Chats Found"),
                (col2, f"{msg_count:,}", "Total Messages"),
                (col3, str(len(agents)), "Agents Detected"),
                (col4, str(min(sample_size, len(chats))), "Analysis Sample"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="val">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True
                    )

            st.success(f"✅ Extracted **{len(chats):,}** chat records. Head to **Build Schema**.")

            with st.expander("👁️ Preview first 3 chats (raw JSON)"):
                for c in chats[:3]:
                    st.json(c, expanded=False)


# ──────────────────────────────────────────────────────────────
# TAB 2 — Build Schema
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-header">Generate AI-derived category schema</div>', unsafe_allow_html=True)

    if "chats" not in st.session_state:
        st.info("👆 Upload a ZIP file in the Upload & Extract tab first.")
    elif not api_key:
        st.warning(f"🔑 Add your {provider} API key in the sidebar.")
    else:
        chats = st.session_state["chats"]
        n = min(sample_size, len(chats))
        st.markdown(
            f"**{provider} · {model_name}** will analyse **{n}** of your **{len(chats):,}** chats "
            f"and produce a clean schema. PIN verification and procedural steps will be excluded."
        )

        if st.button("🚀 Build Schema", type="primary", use_container_width=True):
            sample = random.sample(chats, n)
            with st.spinner("💬 Flattening transcripts…"):
                flat_samples = [flatten_chat(c) for c in sample]

            prompt = build_schema_prompt(flat_samples)
            progress = st.progress(0, f"Sending {n} chats to {provider}…")

            try:
                progress.progress(25, f"{provider} is reading the chats…")
                raw = call_ai(prompt, max_tokens=8192, temperature=0.2)
                progress.progress(80, "Parsing response…")
                schema = parse_json_response(raw)
                st.session_state["schema"] = schema
                progress.progress(100, "Done!")
                time.sleep(0.3)
                progress.empty()
                st.success("✅ Schema built successfully!")

            except json.JSONDecodeError as e:
                progress.empty()
                st.error(f"AI returned non-JSON output. Try again or switch model.\n\n`{e}`")
                with st.expander("Raw response"):
                    st.code(raw)
            except Exception as e:
                progress.empty()
                st.error(f"{provider} error: {e}")

        if "schema" in st.session_state:
            schema = st.session_state["schema"]
            categories = schema.get("categories", [])
            meta = schema.get("meta", {})

            st.markdown("---")

            if meta:
                with st.expander("📋 Analyst Notes", expanded=True):
                    st.markdown(f"> {meta.get('analyst_notes', 'N/A')}")
                    st.caption(
                        f"Schema v{meta.get('schema_version','?')} · "
                        f"derived from {meta.get('derived_from_sample_size','?')} chats"
                    )

            st.markdown(f"### 📂 {len(categories)} Categories Identified")

            for cat in categories:
                with st.expander(f"**{cat.get('name')}**"):
                    st.markdown(f"**Description:** {cat.get('description','')}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**✅ Trigger signals:**")
                        for s in cat.get("signals", []):
                            st.markdown(f"- `{s}`")
                    with col_b:
                        st.markdown("**🚫 Exclusions:**")
                        for e in cat.get("exclude", []):
                            st.markdown(f"- `{e}`")
                    if cat.get("example_intents"):
                        st.markdown("**Example customer phrases:**")
                        for ex in cat["example_intents"]:
                            st.markdown(f"*\"{ex}\"*")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Schema (JSON)",
                    data=json.dumps(schema, indent=2, ensure_ascii=False),
                    file_name="chat_category_schema.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col2:
                rows = []
                for cat in categories:
                    for sig in cat.get("signals", []):
                        rows.append({"category": cat["name"], "signal": sig, "type": "include"})
                    for exc in cat.get("exclude", []):
                        rows.append({"category": cat["name"], "signal": exc, "type": "exclude"})
                st.download_button(
                    "⬇️ Download Signals (CSV)",
                    data=pd.DataFrame(rows).to_csv(index=False),
                    file_name="chat_signals.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ──────────────────────────────────────────────────────────────
# TAB 3 — Re-label chats
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-header">Re-label your chats using the new schema</div>', unsafe_allow_html=True)

    if "chats" not in st.session_state:
        st.info("Upload a ZIP first.")
    elif "schema" not in st.session_state:
        st.info("Build the schema first in the Build Schema tab.")
    elif not api_key:
        st.warning(f"🔑 Add your {provider} API key in the sidebar.")
    else:
        chats = st.session_state["chats"]
        schema = st.session_state["schema"]

        relabel_count = st.slider(
            "How many chats to re-label?",
            min_value=5, max_value=min(200, len(chats)), value=min(50, len(chats))
        )

        if provider == "OpenAI":
            hint = f"gpt-4o-mini recommended for cost. ~{relabel_count * 2}–{relabel_count * 3}s"
        else:
            hint = f"Flash models are fastest. ~{relabel_count * 2}–{relabel_count * 4}s. Watch rate limits."
        st.info(f"⚡ {relabel_count} API calls · {hint}")

        if st.button("🏷️ Re-label Chats", type="primary", use_container_width=True):
            sample = random.sample(chats, relabel_count)
            results = []
            progress = st.progress(0, "Starting…")

            for i, chat in enumerate(sample):
                progress.progress((i + 1) / relabel_count, f"Classifying chat {i+1}/{relabel_count}…")
                flat = flatten_chat(chat)
                prompt = build_relabelling_prompt(schema, flat)

                try:
                    raw = call_ai(prompt, max_tokens=256, temperature=0.1)
                    result = parse_json_response(raw)
                    result["chat_id"] = chat.get("id", i)
                    result["preview"] = flat[:150]
                    results.append(result)
                except Exception as e:
                    results.append({
                        "chat_id": chat.get("id", i),
                        "category": "Error",
                        "confidence": 0,
                        "reason": str(e),
                        "preview": "",
                    })

                if provider == "Gemini":
                    time.sleep(0.3)  # gentle rate-limit buffer

            progress.empty()
            st.session_state["relabelled"] = results
            st.success(f"✅ Classified {len(results)} chats!")

        if "relabelled" in st.session_state:
            results = st.session_state["relabelled"]
            df = pd.DataFrame(results)

            cat_counts = (
                df[df["category"] != "Error"]["category"]
                .value_counts().reset_index()
            )
            cat_counts.columns = ["Category", "Count"]

            st.markdown("#### Distribution of Re-labelled Chats")
            st.bar_chart(cat_counts.set_index("Category"))

            st.markdown("#### Re-labelled Records")
            st.dataframe(
                df[["chat_id", "category", "confidence", "reason", "preview"]],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download Re-labelled CSV",
                data=df.to_csv(index=False),
                file_name="relabelled_chats.csv",
                mime="text/csv",
                use_container_width=True,
            )
