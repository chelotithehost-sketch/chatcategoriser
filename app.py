import streamlit as st
import zipfile
import json
import io
import os
import re
import time
import pandas as pd
from pathlib import Path
import google.generativeai as genai

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

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Header strip */
.app-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b262c 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
}
.app-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: #fff; }
.app-header p  { margin: 0; font-size: 0.9rem; color: #94a3b8; }

/* Metric cards */
.metric-card {
    background: #1e2130; border-radius: 10px;
    padding: 1rem 1.25rem; text-align: center;
    border: 1px solid #2d3349;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }

/* Code / JSON blocks */
.stCodeBlock { font-family: 'DM Mono', monospace !important; }

/* Upload area */
.uploadedFile { border: 1px solid #38bdf8 !important; }

/* Section headers */
.sec-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #1e2130; padding-bottom: .4rem;
    margin: 1.5rem 0 .8rem;
}
</style>
""", unsafe_allow_html=True)

# ── API Key — secrets first, sidebar fallback ─────────────────
_secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    if _secret_key:
        st.success("🔒 Key loaded from Streamlit Secrets")
        api_key = _secret_key
    else:
        api_key = st.text_input(
            "Paste your key",
            type="password",
            placeholder="AIza...",
            help="Or set GEMINI_API_KEY in Streamlit Cloud Secrets to avoid entering it here."
        )

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox("Gemini model", [
        "gemini-3.1-pro-preview",
        "gemini-3-pro",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
    ])
    sample_size = st.slider(
        "Chats to analyse (sample)",
        min_value=10, max_value=500, value=100, step=10,
        help="Gemini analyses this many chats to build the category schema."
    )
    min_category_chats = st.slider(
        "Min chats per category",
        min_value=1, max_value=20, value=3,
        help="Categories with fewer than this many matches are merged into General."
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app extracts tawk.to JSON chats from nested ZIPs, "
        "sends a representative sample to Gemini, and returns a **data-driven "
        "category schema** — free from procedural noise like routine PIN checks."
    )

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <h1>🧠 AI Chat Categoriser</h1>
    <p>Upload your tawk.to ZIP export → get a clean, AI-derived issue taxonomy</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_jsons_from_zip(uploaded_file) -> list[dict]:
    """Recursively dig through nested ZIPs and collect all JSON objects."""
    results = []

    def _process_zip_bytes(zip_bytes: bytes, depth: int = 0):
        if depth > 10:
            return
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    lower = name.lower()
                    data = zf.read(name)
                    if lower.endswith(".zip"):
                        _process_zip_bytes(data, depth + 1)
                    elif lower.endswith(".json"):
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

    _process_zip_bytes(uploaded_file.read())
    return results


def flatten_chat(chat: dict) -> str:
    """Turn a tawk.to chat dict into a readable conversation string."""
    lines = []
    # Metadata
    if chat.get("id"):
        lines.append(f"[ID: {chat['id']}]")
    
    # Messages array (tawk.to structure)
    messages = chat.get("messages", chat.get("conversation", []))
    for msg in messages:
        sender = msg.get("name") or msg.get("sender") or msg.get("type", "?")
        text = msg.get("msg") or msg.get("message") or msg.get("text", "")
        if text:
            lines.append(f"{sender}: {text}")
    
    # Fallback: just dump the whole thing
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

RULES:
1. Categories must reflect genuine customer problems, NOT internal procedures.
   - "Support PIN" or "security PIN" requests are ROUTINE authentication steps 
     performed at the START of nearly every chat — they are NOT a customer issue.
     Do NOT create a category for them.
2. Create 10–22 categories max. Merge thin categories.
3. Each category must have:
   - "name": short, clear title (e.g. "Email Delivery Failures")
   - "description": 1–2 sentences explaining what belongs here
   - "signals": list of 8–20 keyword/phrase triggers (lowercase)
   - "exclude": list of 3–8 phrases that look similar but should NOT trigger this category
   - "example_intents": 2–3 example customer sentences that belong here
4. After the schema, include a "meta" key with:
   - "schema_version": "1.0"
   - "derived_from_sample_size": (integer)
   - "analyst_notes": brief paragraph on what patterns dominated the chats
     and what the old keyword-only approach was getting wrong.

Return ONLY a valid JSON object. No preamble, no markdown fences.

=== CHAT SAMPLE ===
---CHAT---
{sample_block}
"""


def build_relabelling_prompt(schema: dict, chat_text: str) -> str:
    categories = [c["name"] for c in schema.get("categories", [])]
    cat_list = "\n".join(f"- {c}" for c in categories)
    return f"""
You are a support chat classifier. Use the schema below to classify this chat.

AVAILABLE CATEGORIES:
{cat_list}
- General (use only if truly nothing matches)

CHAT:
{chat_text[:1500]}

Return JSON only:
{{"category": "<category name>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}
"""


# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "🧠 Build Schema", "🏷️ Re-label Chats"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — Upload
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-header">Upload your tawk.to ZIP export</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Drop a ZIP file here (nested ZIPs are handled automatically)",
        type=["zip"],
        help="The app will recursively unpack every nested ZIP and collect all JSON files."
    )
    
    if uploaded:
        with st.spinner("🔍 Unpacking ZIPs and reading JSON files…"):
            chats = extract_jsons_from_zip(uploaded)
        
        if not chats:
            st.error("No JSON chat records found in the ZIP. Check the file structure.")
        else:
            st.session_state["chats"] = chats
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="val">{len(chats):,}</div><div class="lbl">Chats Found</div></div>', unsafe_allow_html=True)
            
            msg_count = sum(len(c.get("messages", c.get("conversation", []))) for c in chats)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="val">{msg_count:,}</div><div class="lbl">Total Messages</div></div>', unsafe_allow_html=True)
            
            # Unique agents
            agents = set()
            for c in chats:
                for m in c.get("messages", c.get("conversation", [])):
                    if m.get("type") in ("agent", "staff"):
                        agents.add(m.get("name", ""))
            with col3:
                st.markdown(f'<div class="metric-card"><div class="val">{len(agents)}</div><div class="lbl">Agents Detected</div></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="metric-card"><div class="val">{min(sample_size, len(chats))}</div><div class="lbl">Analysis Sample</div></div>', unsafe_allow_html=True)
            
            st.success(f"✅ Extracted **{len(chats):,}** chat records. Head to **Build Schema** to analyse them.")
            
            # Preview
            with st.expander("👁️ Preview first 3 chats (raw JSON)"):
                for i, c in enumerate(chats[:3]):
                    st.json(c, expanded=False)

# ──────────────────────────────────────────────────────────────
# TAB 2 — Build Schema
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-header">Generate AI-derived category schema</div>', unsafe_allow_html=True)
    
    if "chats" not in st.session_state:
        st.info("👆 Upload a ZIP file in the **Upload & Extract** tab first.")
    elif not api_key:
        st.warning("🔑 Add your Gemini API key in the sidebar.")
    else:
        chats = st.session_state["chats"]
        
        st.markdown(f"""
        Gemini will analyse **{min(sample_size, len(chats))}** of your **{len(chats):,}** chats 
        and produce a clean category schema, automatically excluding procedural noise 
        (like routine PIN verification).
        """)
        
        if st.button("🚀 Build Schema with Gemini", type="primary", use_container_width=True):
            if not api_key:
                st.error("API key required.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                # Sample & flatten
                import random
                sample = random.sample(chats, min(sample_size, len(chats)))
                
                with st.spinner("💬 Flattening chat transcripts…"):
                    flat_samples = [flatten_chat(c) for c in sample]
                
                prompt = build_schema_prompt(flat_samples)
                
                progress = st.progress(0, "Sending to Gemini…")
                try:
                    progress.progress(30, "Gemini is reading the chats…")
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=8192,
                        )
                    )
                    progress.progress(80, "Parsing response…")
                    
                    raw = response.text.strip()
                    # Strip markdown fences if Gemini added them
                    raw = re.sub(r"^```json\s*", "", raw)
                    raw = re.sub(r"^```\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                    
                    schema = json.loads(raw)
                    st.session_state["schema"] = schema
                    progress.progress(100, "Done!")
                    time.sleep(0.3)
                    progress.empty()
                    
                    st.success("✅ Schema built successfully!")
                    
                except json.JSONDecodeError as e:
                    progress.empty()
                    st.error(f"Gemini returned non-JSON output. Try again or switch to a Pro model.\n\n`{e}`")
                    with st.expander("Raw response"):
                        st.code(response.text)
                except Exception as e:
                    progress.empty()
                    st.error(f"Gemini error: {e}")
        
        # Display schema
        if "schema" in st.session_state:
            schema = st.session_state["schema"]
            categories = schema.get("categories", [])
            meta = schema.get("meta", {})
            
            st.markdown("---")
            
            # Meta summary
            if meta:
                with st.expander("📋 Analyst Notes from Gemini", expanded=True):
                    st.markdown(f"> {meta.get('analyst_notes', 'N/A')}")
                    st.caption(f"Schema v{meta.get('schema_version','?')} · derived from {meta.get('derived_from_sample_size','?')} chats")
            
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
            
            # Downloads
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Download Schema (JSON)",
                    data=schema_json,
                    file_name="chat_category_schema.json",
                    mime="application/json",
                    use_container_width=True,
                )
            
            with col2:
                # Flat CSV of signals
                rows = []
                for cat in categories:
                    for sig in cat.get("signals", []):
                        rows.append({"category": cat["name"], "signal": sig, "type": "include"})
                    for exc in cat.get("exclude", []):
                        rows.append({"category": cat["name"], "signal": exc, "type": "exclude"})
                df_signals = pd.DataFrame(rows)
                st.download_button(
                    "⬇️ Download Signals (CSV)",
                    data=df_signals.to_csv(index=False),
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
        st.info("Build the schema first in the **Build Schema** tab.")
    elif not api_key:
        st.warning("🔑 Add your Gemini API key in the sidebar.")
    else:
        chats = st.session_state["chats"]
        schema = st.session_state["schema"]
        
        relabel_count = st.slider(
            "How many chats to re-label?",
            min_value=5, max_value=min(200, len(chats)), value=min(50, len(chats))
        )
        
        st.warning(f"⚡ This will make **{relabel_count}** individual Gemini API calls. "
                   f"Use Flash model for speed/cost. Estimated time: ~{relabel_count*2}–{relabel_count*4}s")
        
        if st.button("🏷️ Re-label Chats", type="primary", use_container_width=True):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            
            import random
            sample = random.sample(chats, relabel_count)
            
            results = []
            progress = st.progress(0, "Starting…")
            
            for i, chat in enumerate(sample):
                progress.progress((i+1)/relabel_count, f"Classifying chat {i+1}/{relabel_count}…")
                flat = flatten_chat(chat)
                prompt = build_relabelling_prompt(schema, flat)
                
                try:
                    resp = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=256)
                    )
                    raw = resp.text.strip()
                    raw = re.sub(r"^```json\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                    result = json.loads(raw)
                    result["chat_id"] = chat.get("id", i)
                    result["preview"] = flat[:150]
                    results.append(result)
                except Exception as e:
                    results.append({
                        "chat_id": chat.get("id", i),
                        "category": "Error",
                        "confidence": 0,
                        "reason": str(e),
                        "preview": ""
                    })
                
                time.sleep(0.1)  # gentle rate limiting
            
            progress.empty()
            st.session_state["relabelled"] = results
            st.success(f"✅ Classified {len(results)} chats!")
        
        if "relabelled" in st.session_state:
            results = st.session_state["relabelled"]
            df = pd.DataFrame(results)
            
            # Summary chart
            cat_counts = df[df["category"] != "Error"]["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            
            st.markdown("#### Distribution of Re-labelled Chats")
            st.bar_chart(cat_counts.set_index("Category"))
            
            # Table
            st.markdown("#### Re-labelled Records")
            st.dataframe(
                df[["chat_id","category","confidence","reason","preview"]],
                use_container_width=True,
                hide_index=True,
            )
            
            # Download
            st.download_button(
                "⬇️ Download Re-labelled CSV",
                data=df.to_csv(index=False),
                file_name="relabelled_chats.csv",
                mime="text/csv",
                use_container_width=True,
            )
