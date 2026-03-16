import streamlit as st
import zipfile
import json
import io
import re
import time
import random
import math
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
.badge-groq   { background: #f5620022; color: #f56200; border: 1px solid #f5620055; }
.sec-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid #1e2130; padding-bottom: .4rem; margin: 1.5rem 0 .8rem;
}
.cost-box {
    background: #0f2a1a; border: 1px solid #10a37f44; border-radius: 8px;
    padding: .75rem 1rem; margin: .5rem 0;
}
.cost-box .cost-val { font-size: 1.4rem; font-weight: 700; color: #10a37f; }
.cost-box .cost-lbl { font-size: 0.8rem; color: #64748b; }
.error-quota { background:#2a0f0f; border:1px solid #ef444444; border-radius:8px; padding:.75rem 1rem; color:#fca5a5; }
.error-rate  { background:#2a1f0f; border:1px solid #f97316aa; border-radius:8px; padding:.75rem 1rem; color:#fdba74; }
.error-auth  { background:#2a1a0f; border:1px solid #eab30844; border-radius:8px; padding:.75rem 1rem; color:#fde68a; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Pricing per 1K tokens (input, output) in USD. Groq free tier = $0.
PRICING = {
    # OpenAI
    "gpt-4o-mini":          (0.00015,  0.00060),
    "gpt-4o":               (0.00500,  0.01500),
    "gpt-4-turbo":          (0.01000,  0.03000),
    "gpt-3.5-turbo":        (0.00050,  0.00150),
    # Gemini
    "gemini-2.5-flash":     (0.000075, 0.00030),
    "gemini-2.5-flash-lite":(0.000025, 0.00010),
    "gemma-3-27b":          (0.0,      0.0),
    "gemini-3-flash":       (0.000075, 0.00030),
    # Groq — free tier (pay-as-you-go is very cheap, treat as ~$0 for free users)
    "llama-3.3-70b-versatile":       (0.0, 0.0),
    "llama-3.1-8b-instant":          (0.0, 0.0),
    "mixtral-8x7b-32768":            (0.0, 0.0),
    "gemma2-9b-it":                  (0.0, 0.0),
}

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality on Groq
    "llama-3.1-8b-instant",      # fastest
    "mixtral-8x7b-32768",        # large context
    "gemma2-9b-it",              # Google Gemma via Groq
]

BATCH_SIZE = 20  # chats per re-label call


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


with st.sidebar:
    st.markdown("### 🤖 AI Provider")
    provider = st.radio("Choose provider", ["Groq", "OpenAI", "Gemini"], horizontal=True,
                        help="Groq is free-tier friendly with fast inference. OpenAI is most reliable. Gemini has rate limits.")
    st.markdown("---")

    # ── Groq ──────────────────────────────────────────────────
    if provider == "Groq":
        st.markdown("### 🔑 Groq API Key")
        _groq_secret = _get_secret("GROQ_API_KEY")
        if _groq_secret:
            st.success("🔒 Key loaded from Streamlit Secrets")
            api_key = _groq_secret
        else:
            api_key = st.text_input("Paste your key", type="password", placeholder="gsk_...",
                                    help="Get a free key at console.groq.com. Set GROQ_API_KEY in Streamlit Secrets.")
        st.markdown("### ⚙️ Model")
        model_name = st.selectbox("Groq model", GROQ_MODELS,
                                  help="llama-3.3-70b-versatile gives best results. llama-3.1-8b-instant is fastest.")
        st.info("💡 Groq free tier is generous — great for testing with no billing needed.")

    # ── OpenAI ────────────────────────────────────────────────
    elif provider == "OpenAI":
        st.markdown("### 🔑 OpenAI API Key")
        _oai_secret = _get_secret("OPENAI_API_KEY")
        if _oai_secret:
            st.success("🔒 Key loaded from Streamlit Secrets")
            api_key = _oai_secret
        else:
            api_key = st.text_input("Paste your key", type="password", placeholder="sk-...",
                                    help="Set OPENAI_API_KEY in Streamlit Cloud Secrets.")
        st.markdown("### ⚙️ Model")
        model_name = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                                  help="gpt-4o-mini is cheapest — costs under $0.01 for 100 chats.")

    # ── Gemini ────────────────────────────────────────────────
    else:
        st.markdown("### 🔑 Gemini API Key")
        _gem_secret = _get_secret("GEMINI_API_KEY")
        if _gem_secret:
            st.success("🔒 Key loaded from Streamlit Secrets")
            api_key = _gem_secret
        else:
            api_key = st.text_input("Paste your key", type="password", placeholder="AIza...",
                                    help="Set GEMINI_API_KEY in Streamlit Cloud Secrets.")
        st.markdown("### ⚙️ Model")
        model_name = st.selectbox("Gemini model",
                                  ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemma-3-27b", "gemini-3-flash"])
        st.warning("⚠️ Gemini free tier has strict rate limits. Use Groq or OpenAI if hitting errors.")

    st.markdown("---")
    st.markdown("### 📊 Analysis Settings")
    sample_size = st.slider(
        "Chats to analyse (sample)", 10, 500, 50, 10,
        help="50–100 chats is enough for a good schema. More doesn't improve quality, just costs more."
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Extracts tawk.to JSON from nested ZIPs, sends a sample to AI, "
        "and returns a clean category schema — free from procedural noise like PIN checks."
    )


# ── Header ────────────────────────────────────────────────────
badge_class = {"OpenAI": "badge-openai", "Gemini": "badge-gemini", "Groq": "badge-groq"}[provider]
badge_html = f'<span class="provider-badge {badge_class}">{provider}</span>'

st.markdown(f"""
<div class="app-header">
  <div>
    <h1>🧠 AI Chat Categoriser &nbsp;{badge_html}</h1>
    <p>Upload your tawk.to ZIP export → get a clean, AI-derived issue taxonomy</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TOKEN & COST UTILITIES
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    in_rate, out_rate = PRICING.get(model, (0.001, 0.002))
    return (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)

def format_cost(usd: float) -> str:
    if usd < 0.0001:
        return "FREE / < $0.0001"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.3f}"


# ═══════════════════════════════════════════════════════════════
# AI CALL ABSTRACTION — OpenAI / Groq share the same client
# (Groq is OpenAI-compatible, just different base_url + no json_object mode)
# ═══════════════════════════════════════════════════════════════

def call_ai(prompt: str, max_tokens: int = 8192, temperature: float = 0.2) -> str:
    """Unified AI call. Returns raw text response."""

    # ── Groq (OpenAI-compatible SDK, different base URL) ──────
    if provider == "Groq":
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            # Note: Groq doesn't support response_format json_object on all models,
            # so we rely on prompt instruction + parse_json_response cleaning instead
        )
        return response.choices[0].message.content

    # ── OpenAI ────────────────────────────────────────────────
    elif provider == "OpenAI":
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

    # ── Gemini ────────────────────────────────────────────────
    else:
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


def parse_json_response(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON — handles both dict and list responses."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Some models wrap arrays: {"results": [...]} — unwrap if needed downstream
    return json.loads(raw)


def friendly_error(e: Exception) -> tuple[str, str]:
    msg = str(e)
    if "insufficient_quota" in msg or ("429" in msg and "quota" in msg.lower()):
        return "error-quota", (
            "🚫 **Quota exceeded** — your OpenAI account has no credit balance.  \n"
            "**Fix:** Go to [platform.openai.com/billing](https://platform.openai.com/billing) "
            "and add at least $5. Running 100 chats costs under **$0.01** with gpt-4o-mini.  \n"
            "**Or switch to Groq** — it's free."
        )
    if "rate_limit" in msg or ("429" in msg and "rate" in msg.lower()):
        return "error-rate", (
            "⏱️ **Rate limit hit** — too many requests too fast.  \n"
            "**Fix:** Wait 30–60 seconds and retry, or lower the sample size. "
            "**Groq** has much more generous rate limits than Gemini."
        )
    if "invalid_api_key" in msg or "401" in msg or "403" in msg:
        return "error-auth", (
            "🔑 **Invalid API key** — the key was rejected.  \n"
            "**Fix:** Check the key starts with `sk-` (OpenAI), `AIza` (Gemini), or `gsk_` (Groq). "
            "Make sure you copied the full key with no trailing spaces."
        )
    return "error-quota", f"**Unexpected error:** `{msg}`"


# ═══════════════════════════════════════════════════════════════
# CHAT HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_jsons_from_zip(uploaded_file) -> list[dict]:
    """Recursively unpack nested ZIPs and collect all JSON objects."""
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


def flatten_chat(chat: dict, max_chars: int = 600) -> str:
    """Flatten a tawk.to chat dict to a compact readable string."""
    lines = []
    if chat.get("id"):
        lines.append(f"[{chat['id']}]")
    messages = chat.get("messages", chat.get("conversation", []))
    for msg in messages:
        sender = msg.get("name") or msg.get("sender") or msg.get("type", "?")
        text = msg.get("msg") or msg.get("message") or msg.get("text", "")
        if text:
            lines.append(f"{sender}: {text[:200]}")
    if len(lines) <= 1:
        lines.append(json.dumps(chat, ensure_ascii=False)[:max_chars])
    return "\n".join(lines)[:max_chars]


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

def build_schema_prompt(chat_samples: list[str]) -> str:
    trimmed = [s[:500] for s in chat_samples]
    sample_block = "\n---\n".join(trimmed)[:80_000]
    return f"""You are an expert customer support analyst at HostAfrica (web hosting company).
Analyse the support chat sample below and produce a CLEAN, DATA-DRIVEN categorisation schema.

RULES:
1. Categories = genuine customer problems ONLY. NOT internal procedures.
   - Support PIN / security PIN = routine auth step performed at the start of almost every chat.
     It is NOT a customer issue. Do NOT create a category for it.
   - No categories for greetings, farewells, or scripted agent steps.
2. 10–20 categories max. Merge thin or overlapping ones.
3. Each category must have:
   - "name": short clear title (e.g. "Email Delivery Failures")
   - "description": 1–2 sentences on what belongs here
   - "signals": 8–20 lowercase keyword/phrase triggers
   - "exclude": 3–8 false-positive phrases to ignore
   - "example_intents": 2–3 example sentences a customer might say
4. Include a "meta" object with:
   - "schema_version": "1.0"
   - "derived_from_sample_size": {len(chat_samples)}
   - "analyst_notes": paragraph on dominant patterns and what the old keyword approach got wrong

Return ONLY valid JSON with keys "categories" and "meta". No markdown, no preamble.

=== CHAT SAMPLE ===
{sample_block}"""


def build_batch_relabel_prompt(schema: dict, chat_batch: list[tuple[int, str]]) -> str:
    """Classify multiple chats in one API call — ~80% cheaper than one-per-chat."""
    categories = [c["name"] for c in schema.get("categories", [])] + ["General"]
    cat_list = "\n".join(f"- {c}" for c in categories)
    chats_block = "\n".join(f"[CHAT_{idx}]\n{text[:400]}" for idx, text in chat_batch)

    return f"""Classify each chat below using ONLY these categories:
{cat_list}

Return a JSON array with exactly {len(chat_batch)} objects.
Each object: {{"id": <chat index 0-based>, "category": "<name>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}

Return ONLY a valid JSON array. No preamble, no markdown.

{chats_block}"""


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "🧠 Build Schema", "🏷️ Re-label Chats"])


# ──────────────────────────────────────────────────────────────
# TAB 1 — Upload & Extract
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-header">Upload your tawk.to ZIP export</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a ZIP file here — nested ZIPs are handled automatically",
        type=["zip"],
        help="The app recursively unpacks every nested ZIP and collects all JSON files."
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

        # Live cost estimate
        sample_preview = random.sample(chats, n)
        flat_preview = [flatten_chat(c) for c in sample_preview]
        prompt_preview = build_schema_prompt(flat_preview)
        est_in  = estimate_tokens(prompt_preview)
        est_out = 3000
        est_cost = estimate_cost(est_in, est_out, model_name)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f'<div class="cost-box"><div class="cost-val">{n}</div>'
                        f'<div class="cost-lbl">Chats in sample</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="cost-box"><div class="cost-val">~{est_in:,}</div>'
                        f'<div class="cost-lbl">Estimated input tokens</div></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="cost-box"><div class="cost-val">{format_cost(est_cost)}</div>'
                        f'<div class="cost-lbl">Estimated cost (USD)</div></div>', unsafe_allow_html=True)

        st.caption(f"Using **{provider} · {model_name}**. Adjust sample size in the sidebar to control cost.")
        st.markdown("---")

        if st.button("🚀 Build Schema", type="primary", use_container_width=True):
            prompt = build_schema_prompt(flat_preview)
            progress = st.progress(0, f"Sending {n} chats to {provider}…")
            raw = ""
            try:
                progress.progress(25, f"{provider} is analysing the chats…")
                raw = call_ai(prompt, max_tokens=4096, temperature=0.2)
                progress.progress(80, "Parsing response…")
                schema = parse_json_response(raw)
                if not isinstance(schema, dict) or "categories" not in schema:
                    raise ValueError("Response missing 'categories' key.")
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
                css_cls, msg = friendly_error(e)
                st.markdown(f'<div class="{css_cls}">{msg}</div>', unsafe_allow_html=True)

        # Display schema
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
# TAB 3 — Re-label (batched)
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
            min_value=5, max_value=min(500, len(chats)), value=min(100, len(chats))
        )

        # Batched cost estimate
        n_batches = math.ceil(relabel_count / BATCH_SIZE)
        sample_rl = random.sample(chats, min(BATCH_SIZE, relabel_count))
        flat_rl = [(i, flatten_chat(c)) for i, c in enumerate(sample_rl)]
        batch_ex = build_batch_relabel_prompt(schema, flat_rl)
        tpb = estimate_tokens(batch_ex)
        total_in  = tpb * n_batches
        total_out = 150 * n_batches
        cost_rl = estimate_cost(total_in, total_out, model_name)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f'<div class="cost-box"><div class="cost-val">{n_batches}</div>'
                        f'<div class="cost-lbl">API calls (batched {BATCH_SIZE}/call)</div></div>',
                        unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="cost-box"><div class="cost-val">~{total_in:,}</div>'
                        f'<div class="cost-lbl">Estimated input tokens</div></div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f'<div class="cost-box"><div class="cost-val">{format_cost(cost_rl)}</div>'
                        f'<div class="cost-lbl">Estimated cost (USD)</div></div>', unsafe_allow_html=True)

        st.caption(f"✅ Batching {BATCH_SIZE} chats per call — ~80% cheaper than one-per-chat.")

        if st.button("🏷️ Re-label Chats", type="primary", use_container_width=True):
            sample = random.sample(chats, relabel_count)
            indexed = [(i, flatten_chat(c)) for i, c in enumerate(sample)]
            batches = [indexed[i:i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]

            results = []
            progress = st.progress(0, "Starting…")
            errors = 0

            for b_idx, batch in enumerate(batches):
                progress.progress(
                    (b_idx + 1) / len(batches),
                    f"Batch {b_idx+1}/{len(batches)} — classifying {len(batch)} chats…"
                )
                prompt = build_batch_relabel_prompt(schema, batch)
                raw_clean = ""
                try:
                    raw = call_ai(prompt, max_tokens=1024, temperature=0.1)
                    raw_clean = raw.strip()
                    raw_clean = re.sub(r"^```json\s*", "", raw_clean)
                    raw_clean = re.sub(r"^```\s*", "", raw_clean)
                    raw_clean = re.sub(r"\s*```$", "", raw_clean)

                    batch_results = json.loads(raw_clean)

                    # Unwrap if model returned {"results": [...]} instead of [...]
                    if isinstance(batch_results, dict):
                        batch_results = next(
                            (v for v in batch_results.values() if isinstance(v, list)), []
                        )

                    for item in batch_results:
                        orig_idx = item.get("id", 0)
                        if 0 <= orig_idx < len(batch):
                            chat_obj = sample[batch[orig_idx][0]]
                            results.append({
                                "chat_id":    chat_obj.get("id", batch[orig_idx][0]),
                                "category":   item.get("category", "General"),
                                "confidence": round(float(item.get("confidence", 0)), 2),
                                "reason":     item.get("reason", ""),
                                "preview":    batch[orig_idx][1][:150],
                            })

                except Exception as e:
                    errors += 1
                    css_cls, msg = friendly_error(e)
                    st.markdown(
                        f'<div class="{css_cls}">Batch {b_idx+1} failed: {msg}</div>',
                        unsafe_allow_html=True
                    )
                    if "quota" in str(e).lower() or "insufficient" in str(e).lower():
                        break  # stop if quota is exhausted

                # Groq & Gemini benefit from a small pause between batches
                if provider in ("Gemini", "Groq"):
                    time.sleep(0.4)

            progress.empty()

            if results:
                st.session_state["relabelled"] = results
                msg_suffix = f" ({errors} batch(es) failed)" if errors else ""
                st.success(f"✅ Classified {len(results)} chats{msg_suffix}")
            else:
                st.error("No chats were classified. Check the error messages above.")

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
