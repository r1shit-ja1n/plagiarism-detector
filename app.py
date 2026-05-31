"""
═══════════════════════════════════════════════════════════════════════
  PLAGCHECK ENTERPRISE — Streamlit Frontend for C++ Detection Engine
═══════════════════════════════════════════════════════════════════════
"""
import streamlit as st
import subprocess
import tempfile
import os
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlagCheck Enterprise",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# CUSTOM DARK THEME CSS  (SaaS-grade look)
# ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Global background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* Top header banner */
    .hero-banner {
        background: linear-gradient(90deg, #1f6feb 0%, #8957e5 50%, #db61a2 100%);
        padding: 28px 40px;
        border-radius: 14px;
        margin-bottom: 28px;
        box-shadow: 0 10px 40px rgba(31, 111, 235, 0.25);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }

    /* Card-like containers */
    .doc-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .doc-card h3 {
        color: #58a6ff;
        margin-top: 0;
        font-size: 1.2rem;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #1f6feb;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 8px 0;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #e6edf3;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Big score widget */
    .score-display {
        text-align: center;
        padding: 48px 24px;
        border-radius: 18px;
        margin: 24px 0;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .score-display .number {
        font-size: 5.5rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .score-display .label {
        font-size: 1.3rem;
        margin-top: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .score-low    { background: linear-gradient(135deg, #0f5132 0%, #198754 100%); color: #d1e7dd; }
    .score-mod    { background: linear-gradient(135deg, #664d03 0%, #ffc107 100%); color: #1a1a1a; }
    .score-high   { background: linear-gradient(135deg, #842029 0%, #dc3545 100%); color: #ffe5e8; }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] h2 {
        color: #58a6ff;
        font-size: 1.15rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(31, 111, 235, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(31, 111, 235, 0.5);
    }

    /* Section dividers */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #30363d 50%, transparent 100%);
        margin: 32px 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #6e7681;
        font-size: 0.85rem;
        padding: 24px 0 8px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🔍 PlagCheck Enterprise</h1>
    <p>High-performance plagiarism detection · KMP · Rabin-Karp · LCS · N-Gram Jaccard</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Engine Configuration")
    st.caption("These settings are forwarded directly to the C++ binary.")

    st.markdown("### Algorithm")
    algorithm = st.selectbox(
        "Preferred matching algorithm",
        options=["ALL", "KMP", "RK", "LCS"],
        index=0,
        help="ALL = average of KMP, Rabin-Karp, and LCS scores",
    )

    st.markdown("### N-Gram Tuning")
    ngram_size = st.slider("N-gram size", 1, 8, 3,
                           help="Window width used for Jaccard set construction")

    st.markdown("### Decision Thresholds (%)")
    low_threshold, high_threshold = st.slider(
        "Low ↔ High",
        min_value=0, max_value=100, value=(30, 70),
        help="Below LOW → likely original · Above HIGH → likely plagiarized",
    )

    st.markdown("### Preprocessing")
    remove_stopwords = st.toggle("Remove stopwords", value=True)

    with st.expander("ℹ️ Binary Path"):
        binary_path = st.text_input("Executable", value="./plagcheck")
        st.caption("Compile with: `g++ -std=c++17 -O2 -o plagcheck plagiarism.cpp`")

    st.markdown("---")
    st.caption("⚡ Powered by a native C++17 engine")

# ──────────────────────────────────────────────────────────────────────
# DUAL INPUT — DOCUMENT A and DOCUMENT B
# ──────────────────────────────────────────────────────────────────────
st.markdown("### 📑 Documents to Compare")

col_a, col_b = st.columns(2, gap="large")

def document_input(col, label, key_prefix):
    with col:
        st.markdown(f'<div class="doc-card"><h3>{label}</h3>', unsafe_allow_html=True)
        mode = st.radio(
            f"Input mode — {label}",
            ["📁 Upload .txt", "✍️ Paste text"],
            horizontal=True,
            key=f"{key_prefix}_mode",
            label_visibility="collapsed",
        )
        text = ""
        if mode.startswith("📁"):
            up = st.file_uploader(
                f"Upload {label}",
                type=["txt"],
                key=f"{key_prefix}_file",
                label_visibility="collapsed",
            )
            if up is not None:
                text = up.read().decode("utf-8", errors="ignore")
                st.success(f"Loaded {len(text):,} characters from {up.name}")
        else:
            text = st.text_area(
                f"Paste {label}",
                height=240,
                key=f"{key_prefix}_text",
                placeholder=f"Paste the content of {label} here…",
                label_visibility="collapsed",
            )
            if text:
                st.caption(f"📝 {len(text):,} chars · {len(text.split()):,} words")
        st.markdown('</div>', unsafe_allow_html=True)
        return text

text_a = document_input(col_a, "Document A", "a")
text_b = document_input(col_b, "Document B", "b")

# ──────────────────────────────────────────────────────────────────────
# ANALYZE BUTTON
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

centered = st.columns([1, 2, 1])
with centered[1]:
    analyze = st.button("🚀 Analyze Documents", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# SUBPROCESS BRIDGE
# ──────────────────────────────────────────────────────────────────────
def run_engine(file_a: str, file_b: str, cfg: dict, binary: str) -> dict:
    """Invoke the C++ binary and parse its key=value output."""
    cmd = [
        binary,
        file_a,
        file_b,
        str(cfg["ngram"]),
        cfg["algo"],
        str(cfg["low"]),
        str(cfg["high"]),
        "1" if cfg["stopwords"] else "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(proc.stderr or "Engine failed with no output")

    parsed = {}
    for line in proc.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed

def score_color_class(score: float, low: float, high: float) -> tuple:
    if score < low:
        return "score-low", "✓ LIKELY ORIGINAL"
    if score <= high:
        return "score-mod", "⚠ REVIEW RECOMMENDED"
    return "score-high", "✗ LIKELY PLAGIARIZED"

# ──────────────────────────────────────────────────────────────────────
# EXECUTION + RESULTS
# ──────────────────────────────────────────────────────────────────────
if analyze:
    if not text_a.strip() or not text_b.strip():
        st.error("⚠️ Both Document A and Document B must contain text before analysis.")
        st.stop()

    # Write inputs to temp files
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fa:
        fa.write(text_a)
        path_a = fa.name
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fb:
        fb.write(text_b)
        path_b = fb.name

    cfg = {
        "ngram": ngram_size,
        "algo": algorithm,
        "low": low_threshold,
        "high": high_threshold,
        "stopwords": remove_stopwords,
    }

    try:
        with st.spinner("🔬 Engine running… invoking C++ binary"):
            t0 = time.perf_counter()
            results = run_engine(path_a, path_b, cfg, binary_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        overall = float(results.get("OVERALL", 0.0))
        kmp     = float(results.get("KMP", 0.0))
        rk      = float(results.get("RK", 0.0))
        lcs     = float(results.get("LCS", 0.0))
        verdict = results.get("VERDICT", "UNKNOWN")

        # ── Big Score Display ─────────────────────────────────────────
        cls, label = score_color_class(overall, low_threshold, high_threshold)
        st.markdown(f"""
        <div class="score-display {cls}">
            <div class="number">{overall:.2f}%</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Algorithm Breakdown Metric Cards ─────────────────────────
        st.markdown("### 📊 Algorithm Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        for col, name, value, time_us in [
            (m1, "KMP",          kmp, results.get("KMP_TIME_US", "—")),
            (m2, "Rabin-Karp",   rk,  results.get("RK_TIME_US",  "—")),
            (m3, "LCS",          lcs, results.get("LCS_TIME_US", "—")),
            (m4, "Wall Time",    elapsed_ms, None),
        ]:
            with col:
                if name == "Wall Time":
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Latency</div>
                        <div class="metric-value">{value:.1f} ms</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value">{value:.2f}%</div>
                        <div style="color:#8b949e;font-size:0.8rem;margin-top:4px;">
                            engine time: {time_us} µs
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Detailed Diagnostics ─────────────────────────────────────
        with st.expander("🔍 Detailed Engine Diagnostics"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Match Counts**")
                st.write(f"- KMP n-gram hits: `{results.get('KMP_MATCHES', '—')}`")
                st.write(f"- Rabin-Karp n-gram hits: `{results.get('RK_MATCHES', '—')}`")
                st.write(f"- LCS length (tokens): `{results.get('LCS_LEN', '—')}`")
            with d2:
                st.markdown("**Configuration Used**")
                st.write(f"- N-gram size: `{ngram_size}`")
                st.write(f"- Algorithm: `{algorithm}`")
                st.write(f"- Thresholds: `{low_threshold}% / {high_threshold}%`")
                st.write(f"- Stopwords removed: `{remove_stopwords}`")

        with st.expander("📜 Raw C++ Engine Output"):
            st.code("\n".join(f"{k}={v}" for k, v in results.items()), language="ini")

    except FileNotFoundError:
        st.error(f"""
        ### 🛑 C++ Binary Not Found
        Could not locate the executable at **`{binary_path}`**.

        **Fix it by compiling first:**
        ```bash
        g++ -std=c++17 -O2 -o plagcheck plagiarism.cpp
        ```
        Then place `plagcheck` in the same directory as `app.py`, or update the
        binary path in the sidebar.
        """)
    except subprocess.TimeoutExpired:
        st.error("⏱️ Engine timed out (>60s). Try shorter documents or a smaller N-gram.")
    except Exception as e:
        st.error(f"❌ Engine error: `{e}`")
    finally:
        for p in (path_a, path_b):
            try: os.unlink(p)
            except OSError: pass

# ──────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    PlagCheck Enterprise · C++17 backend · Streamlit frontend ·
    KMP / Rabin-Karp / LCS / N-Gram Jaccard
</div>
""", unsafe_allow_html=True)
