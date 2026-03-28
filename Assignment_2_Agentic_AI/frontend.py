from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
OUTPUTS_DIR = ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_research, save_report  # noqa: E402

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #ede9fe;
}

.stApp {
    background:
        radial-gradient(ellipse 900px 500px at 10% 20%, rgba(139, 92, 246, 0.18), transparent),
        radial-gradient(ellipse 800px 400px at 80% 10%, rgba(236, 72, 153, 0.14), transparent),
        radial-gradient(ellipse 600px 600px at 50% 80%, rgba(99, 102, 241, 0.10), transparent),
        linear-gradient(160deg, #0c0a1d 0%, #110e24 40%, #0f0b1e 100%);
}

.main .block-container {
    max-width: 1080px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ───── Hero ───── */
.hero {
    position: relative;
    border-radius: 24px;
    padding: 2.5rem 2.2rem 2rem;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(236, 72, 153, 0.08));
    border: 1px solid rgba(167, 139, 250, 0.22);
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    overflow: hidden;
    margin-bottom: 1.4rem;
    animation: heroSlide 600ms cubic-bezier(0.22, 1, 0.36, 1);
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 180deg, transparent 0deg, rgba(139, 92, 246, 0.06) 60deg, transparent 120deg);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}

.hero h1 {
    position: relative;
    margin: 0 0 0.5rem 0;
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #e9d5ff, #f9a8d4, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    position: relative;
    margin: 0 0 1.2rem 0;
    color: #c4b5fd;
    line-height: 1.6;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 600px;
}

.credit {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: #fdf4ff;
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(236, 72, 153, 0.18));
    border: 1px solid rgba(192, 132, 252, 0.35);
    padding: 0.55rem 1.2rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 500;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.credit:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.25);
}

/* ───── Panels ───── */
.panel {
    border-radius: 20px;
    padding: 1.4rem;
    margin-top: 1.2rem;
    background: rgba(17, 14, 36, 0.72);
    border: 1px solid rgba(167, 139, 250, 0.18);
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    animation: fadeUp 500ms cubic-bezier(0.22, 1, 0.36, 1);
}

.report-card {
    border-radius: 20px;
    padding: 1.4rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    background: linear-gradient(145deg, rgba(17, 14, 36, 0.85), rgba(30, 20, 60, 0.65));
    border: 1px solid rgba(192, 132, 252, 0.2);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    animation: fadeUp 560ms cubic-bezier(0.22, 1, 0.36, 1);
}

/* ───── Section Headers ───── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}

.section-header .icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(236, 72, 153, 0.15));
    border: 1px solid rgba(167, 139, 250, 0.3);
}

.section-header h3 {
    margin: 0;
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e9d5ff;
}

/* ───── Inputs ───── */
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    color: #c4b5fd;
    font-weight: 500;
    font-size: 0.95rem;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(17, 14, 36, 0.9);
    color: #f5f3ff;
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 12px;
    font-size: 0.95rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(168, 85, 247, 0.6);
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15);
}

[data-testid="stTextInput"] input::placeholder {
    color: #7c6fa0;
}

/* ───── Buttons ───── */
[data-testid="stButton"] button {
    border-radius: 12px;
    border: 1px solid rgba(168, 85, 247, 0.5);
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    color: #ffffff;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3);
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #a855f7, #c084fc);
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.45);
    transform: translateY(-1px);
}

[data-testid="stDownloadButton"] button {
    border-radius: 12px;
    border: 1px solid rgba(236, 72, 153, 0.45);
    background: linear-gradient(135deg, #ec4899, #f472b6);
    color: #fff0f6;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(236, 72, 153, 0.25);
}

[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #f472b6, #f9a8d4);
    box-shadow: 0 6px 24px rgba(236, 72, 153, 0.4);
    transform: translateY(-1px);
}

/* ───── Delete Button ───── */
.delete-btn button {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.08)) !important;
    border-color: rgba(239, 68, 68, 0.35) !important;
    color: #fca5a5 !important;
}

.delete-btn button:hover {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.15)) !important;
}

/* ───── Typography ───── */
.caption {
    color: #a78bfa;
    margin-top: 0.25rem;
    font-size: 0.88rem;
    font-weight: 300;
}

h2, h3 {
    margin-top: 0.6rem;
    margin-bottom: 0.4rem;
}

/* ───── Divider ───── */
.stDivider {
    margin: 0.9rem 0 1.1rem;
    border-color: rgba(167, 139, 250, 0.12) !important;
}

/* ───── Stats pill ───── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.2);
    padding: 0.35rem 0.85rem;
    border-radius: 50px;
    font-size: 0.82rem;
    color: #c4b5fd;
    font-weight: 400;
}

/* ───── Animations ───── */
@keyframes heroSlide {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
</style>
""",
    unsafe_allow_html=True,
)

load_dotenv(ROOT / ".env")


def list_recent_reports() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        [p for p in OUTPUTS_DIR.glob("report_*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def hydrate_state() -> None:
    if "report_paths" not in st.session_state:
        st.session_state["report_paths"] = list_recent_reports()
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "report" not in st.session_state:
        st.session_state["report"] = ""
    if "output_path" not in st.session_state:
        st.session_state["output_path"] = ""
    if "scroll_top" not in st.session_state:
        st.session_state["scroll_top"] = False


hydrate_state()

if st.session_state["scroll_top"]:
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
    st.session_state["scroll_top"] = False

# ───── Hero Section ─────
st.markdown(
    """
<div class="hero">
  <h1>🧠 Research Agent</h1>
  <p>Generate comprehensive, AI-powered research reports in seconds.
     Enter any topic and let the autonomous agent handle the rest.</p>
  <div class="credit">✨ Made by Aditi Jha</div>
</div>
""",
    unsafe_allow_html=True,
)

paths: list[Path] = st.session_state["report_paths"]
report_names = [p.name for p in paths] if paths else []

if report_names and "selected_report_name" not in st.session_state:
    st.session_state["selected_report_name"] = report_names[0]

selected_name = st.session_state.get("selected_report_name", report_names[0] if report_names else "")
selected_index = report_names.index(selected_name) if (report_names and selected_name in report_names) else 0

# ───── Stats Row ─────
num_reports = len(paths)
st.markdown(
    f"""
<div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-bottom:1rem;">
  <span class="stat-pill">📄 {num_reports} report{'s' if num_reports != 1 else ''} saved</span>
  <span class="stat-pill">⚡ Powered by Groq LLaMA 3.3</span>
  <span class="stat-pill">🔍 Web + Wikipedia Search</span>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ───── Create Report Section ─────
st.markdown(
    '<div class="section-header"><div class="icon">✍️</div><h3>Create New Report</h3></div>',
    unsafe_allow_html=True,
)

topic = st.text_input(
    "Research Topic",
    value="Impact of AI in Healthcare",
    placeholder="e.g. Quantum Computing, Climate Change, Space Exploration...",
)

col_gen, col_info = st.columns([1, 2])
with col_gen:
    run_clicked = st.button("🚀 Generate Report", use_container_width=True, type="primary")
with col_info:
    st.caption("Your new report will appear at the top automatically.")

if run_clicked:
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("🔬 Researching and drafting report..."):
            try:
                report = run_research(
                    topic=topic.strip(),
                    provider=DEFAULT_PROVIDER,
                    model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )
                output_path = save_report(topic.strip(), report, OUTPUTS_DIR)

                st.session_state["report"] = report
                st.session_state["output_path"] = str(output_path)
                st.session_state["report_paths"] = list_recent_reports()
                st.session_state["selected_report_name"] = output_path.name
                st.session_state["scroll_top"] = True
                st.rerun()
            except Exception:
                st.error("Could not generate the report right now. Please try again.")

st.divider()

# ───── Current Report Section ─────
if paths:
    selected_path = paths[selected_index]
    if selected_path.exists():
        st.session_state["report"] = selected_path.read_text(encoding="utf-8")
        st.session_state["output_path"] = str(selected_path)

if st.session_state["report"]:
    st.markdown(
        '<div class="section-header"><div class="icon">📋</div><h3>Current Report</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="report-card">' + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state["report"])
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=st.session_state["report"],
        file_name=Path(st.session_state["output_path"]).name if st.session_state["output_path"] else "research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()

# ───── Recent Reports Section ─────
st.markdown(
    '<div class="section-header"><div class="icon">📚</div><h3>Recent Reports</h3></div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    selected_name = st.selectbox(
        "Recent Reports",
        options=report_names if report_names else ["No reports yet"],
        index=selected_index if report_names else 0,
        disabled=not report_names,
        key="selected_report_name",
        on_change=lambda: st.session_state.update({"scroll_top": True}),
    )
with col2:
    st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
    delete_clicked = st.button("🗑️ Delete", use_container_width=True, disabled=not report_names, key="delete_btn")
    st.markdown("</div>", unsafe_allow_html=True)

if report_names and delete_clicked:
    to_delete = paths[selected_index]
    try:
        to_delete.unlink(missing_ok=True)
        st.session_state["report_paths"] = list_recent_reports()
        updated_paths = st.session_state["report_paths"]
        if not updated_paths:
            st.session_state["report"] = ""
            st.session_state["output_path"] = ""
            st.session_state["selected_report_name"] = ""
        else:
            next_index = min(selected_index, len(updated_paths) - 1)
            st.session_state["selected_report_name"] = updated_paths[next_index].name
            st.session_state["report"] = updated_paths[next_index].read_text(encoding="utf-8")
            st.session_state["output_path"] = str(updated_paths[next_index])
        st.session_state["scroll_top"] = True
        st.rerun()
    except OSError:
        st.error("Could not delete the selected report.")
