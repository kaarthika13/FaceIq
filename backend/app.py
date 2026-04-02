"""
app.py
------
Hybrid Facial Characterization and Explainable Image Retrieval System
Streamlit frontend — white + orange theme.

Run:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import io

st.set_page_config(
    page_title="FaceIQ — Facial Characterization System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ─── Global Reset ─────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif;
}
.stApp { background: #FFFFFF; }
#MainMenu, footer, header { visibility: hidden; }

/* ══════════════════════════════════════════════════
   HOME PAGE
══════════════════════════════════════════════════ */

.hp-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.4rem 3.5rem;
  border-bottom: 1px solid #F0EDE8;
  background: #FFFFFF;
}
.hp-brand {
  font-family: 'Playfair Display', serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: #1C1C1C;
  letter-spacing: -0.02em;
}
.hp-brand span { color: #F05A00; }
.hp-nav-links {
  display: flex;
  gap: 2.2rem;
  align-items: center;
}
.hp-nav-link {
  font-size: 0.85rem;
  font-weight: 500;
  color: #6B6B6B;
  letter-spacing: 0.01em;
  cursor: default;
}
.hp-nav-tag {
  background: #FFF1E6;
  color: #F05A00;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.22rem 0.65rem;
  border-radius: 4px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  border: 1px solid #FFDEC7;
}

.hp-hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  min-height: 560px;
  border-bottom: 1px solid #F0EDE8;
  overflow: hidden;
}
.hp-hero-left {
  padding: 5rem 3.5rem 4rem;
  background: #FFFFFF;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-right: 1px solid #F0EDE8;
}
.hp-hero-right {
  background: #FFF8F3;
  padding: 3.5rem 3rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  position: relative;
  overflow: hidden;
}
.hp-hero-right::before {
  content: '';
  position: absolute;
  width: 380px; height: 380px;
  border-radius: 50%;
  background: radial-gradient(circle, #FFD4B0 0%, transparent 70%);
  top: -80px; right: -100px;
  opacity: 0.5;
  pointer-events: none;
}
.hp-hero-right::after {
  content: '';
  position: absolute;
  width: 200px; height: 200px;
  border-radius: 50%;
  background: radial-gradient(circle, #FFBA80 0%, transparent 70%);
  bottom: -50px; left: 20px;
  opacity: 0.3;
  pointer-events: none;
}

.hp-eyebrow {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #F05A00;
  margin-bottom: 1.4rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.hp-eyebrow-line {
  width: 28px;
  height: 2px;
  background: #F05A00;
  display: inline-block;
  border-radius: 2px;
}

.hp-headline {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.6rem, 4vw, 3.8rem);
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.025em;
  color: #1C1C1C;
  margin: 0 0 1.4rem;
}
.hp-headline em {
  font-style: italic;
  color: #F05A00;
}

.hp-desc {
  font-size: 1rem;
  color: #6B6B6B;
  line-height: 1.72;
  font-weight: 300;
  max-width: 480px;
  margin-bottom: 2.4rem;
}

.hp-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 2.8rem;
}
.hp-pill {
  background: #FFF1E6;
  border: 1px solid #FFDEC7;
  color: #B84500;
  font-size: 0.76rem;
  font-weight: 500;
  padding: 0.3rem 0.85rem;
  border-radius: 100px;
  letter-spacing: 0.01em;
}

.hp-stats-panel {
  position: relative;
  z-index: 2;
  width: 100%;
}
.hp-stats-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  font-weight: 600;
  color: #1C1C1C;
  margin-bottom: 0.4rem;
  letter-spacing: -0.02em;
}
.hp-stats-title em { font-style: italic; color: #F05A00; }
.hp-stats-sub {
  font-size: 0.82rem;
  color: #9A9A9A;
  font-weight: 300;
  margin-bottom: 2.2rem;
  line-height: 1.6;
}
.hp-stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  background: #EEE8E0;
  border: 1px solid #EEE8E0;
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 1.6rem;
}
.hp-stat-cell {
  background: #FFFFFF;
  padding: 1.2rem 1.4rem;
  transition: background 0.2s;
}
.hp-stat-cell:hover { background: #FFF8F3; }
.hp-stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 600;
  color: #F05A00;
  letter-spacing: -0.04em;
  line-height: 1;
  display: block;
}
.hp-stat-lbl {
  font-size: 0.7rem;
  color: #A0A0A0;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  display: block;
  margin-top: 0.25rem;
}

.hp-section {
  padding: 4rem 3.5rem;
  border-bottom: 1px solid #F0EDE8;
}
.hp-section-header {
  display: flex;
  align-items: baseline;
  gap: 1.2rem;
  margin-bottom: 2.5rem;
}
.hp-section-num {
  font-family: 'Playfair Display', serif;
  font-size: 0.8rem;
  font-weight: 400;
  color: #CCBFB5;
  font-style: italic;
  white-space: nowrap;
}
.hp-section-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  font-weight: 600;
  color: #1C1C1C;
  letter-spacing: -0.02em;
  white-space: nowrap;
}
.hp-section-rule {
  flex: 1;
  height: 1px;
  background: #F0EDE8;
  margin-left: 0.5rem;
  align-self: center;
}

.hp-pipeline-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.2rem;
}
.hp-pipe-card {
  border: 1px solid #EEE8E0;
  border-radius: 14px;
  padding: 1.6rem 1.8rem;
  background: #FFFFFF;
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.hp-pipe-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: #F05A00;
  opacity: 0;
  transition: opacity 0.25s;
}
.hp-pipe-card:hover {
  border-color: #FFDEC7;
  box-shadow: 0 8px 32px rgba(240,90,0,0.08);
  transform: translateY(-2px);
}
.hp-pipe-card:hover::before { opacity: 1; }
.hp-pipe-step {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: #D4C5B8;
  margin-bottom: 0.8rem;
}
.hp-pipe-icon { font-size: 1.5rem; margin-bottom: 0.6rem; display: block; }
.hp-pipe-name {
  font-size: 0.95rem;
  font-weight: 600;
  color: #1C1C1C;
  margin-bottom: 0.35rem;
  letter-spacing: -0.01em;
}
.hp-pipe-desc {
  font-size: 0.78rem;
  color: #8A8A8A;
  line-height: 1.6;
  font-weight: 300;
}

.hp-how-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0;
  background: #FFF8F3;
  border-top: 1px solid #F0EDE8;
  border-bottom: 1px solid #F0EDE8;
}
.hp-how-cell {
  padding: 2.2rem 2rem;
  border-right: 1px solid #F0EDE8;
}
.hp-how-cell:last-child { border-right: none; }
.hp-how-num {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 700;
  color: #FFD4B0;
  line-height: 1;
  margin-bottom: 0.6rem;
  letter-spacing: -0.04em;
}
.hp-how-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: #1C1C1C;
  margin-bottom: 0.4rem;
}
.hp-how-desc {
  font-size: 0.76rem;
  color: #9A9A9A;
  line-height: 1.6;
  font-weight: 300;
}

.hp-cta-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 3rem 3.5rem;
  background: #1C1C1C;
}
.hp-cta-left {
  font-family: 'Playfair Display', serif;
  font-size: 1.75rem;
  font-weight: 400;
  color: #FFFFFF;
  font-style: italic;
  letter-spacing: -0.02em;
  max-width: 520px;
  line-height: 1.3;
}
.hp-cta-left span { color: #FF7A1A; font-style: normal; font-weight: 600; }

.hp-footer {
  padding: 1.4rem 3.5rem;
  background: #1C1C1C;
  border-top: 1px solid #2A2A2A;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hp-footer-left { font-size: 0.76rem; color: #555; font-weight: 300; }
.hp-footer-right { display: flex; gap: 1.8rem; font-size: 0.76rem; color: #555; }

/* Homepage buttons */
.hp-cta-btn .stButton > button {
  background: #F05A00 !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.72rem 2.2rem !important;
  box-shadow: 0 4px 24px rgba(240,90,0,0.28) !important;
  letter-spacing: 0.02em !important;
  transition: all 0.2s !important;
}
.hp-cta-btn .stButton > button:hover {
  background: #D44F00 !important;
  box-shadow: 0 6px 32px rgba(240,90,0,0.42) !important;
  transform: translateY(-1px) !important;
}

.hp-cta-btn-outline .stButton > button {
  background: transparent !important;
  color: #F05A00 !important;
  border: 1.5px solid #F05A00 !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.62rem 1.6rem !important;
  box-shadow: none !important;
}
.hp-cta-btn-outline .stButton > button:hover {
  background: #FFF1E6 !important;
  box-shadow: none !important;
  transform: none !important;
}

.hp-cta-btn-white .stButton > button {
  background: #FFFFFF !important;
  color: #1C1C1C !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.72rem 2.2rem !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
.hp-cta-btn-white .stButton > button:hover {
  background: #F5F5F5 !important;
  box-shadow: 0 6px 28px rgba(0,0,0,0.28) !important;
  transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════════════════
   ANALYSIS PAGE
══════════════════════════════════════════════════ */

.fiq-header {
  text-align: center;
  padding: 3rem 1rem 1.5rem;
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F0 100%);
  border-bottom: 3px solid #F05A00;
}
.fiq-logo {
  font-family: 'Playfair Display', serif;
  font-size: 3rem;
  font-weight: 700;
  color: #F05A00;
  letter-spacing: -0.02em;
  margin: 0;
  line-height: 1;
}
.fiq-logo span { color: #1C1C1C; }
.fiq-subtitle {
  font-size: 1rem;
  color: #6B7280;
  font-weight: 300;
  margin-top: 0.5rem;
}
.fiq-badge {
  display: inline-block;
  background: #F05A00;
  color: white;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-left: 0.5rem;
  vertical-align: middle;
}

.fiq-card {
  background: #FFFFFF;
  border-radius: 14px;
  padding: 1.5rem 1.8rem;
  box-shadow: 0 2px 12px rgba(0,0,0,0.05);
  margin-bottom: 1.2rem;
  border: 1px solid #F0EDE8;
}
.fiq-card-accent { border-left: 3px solid #F05A00; }

.section-title {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #B0A8A0;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #F0EDE8;
}

.stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.stat-chip {
  background: #FFF1E6;
  border: 1.5px solid #FFDEC7;
  border-radius: 10px;
  padding: 0.7rem 1.1rem;
  min-width: 90px;
  text-align: center;
}
.stat-chip .val {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  font-weight: 600;
  color: #F05A00;
  line-height: 1;
  display: block;
}
.stat-chip .lbl {
  font-size: 0.68rem;
  color: #B0A8A0;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  display: block;
  margin-top: 0.2rem;
}

.tag-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.tag {
  background: #FFF1E6;
  color: #B84500;
  border: 1px solid #FFDEC7;
  border-radius: 100px;
  padding: 0.22rem 0.72rem;
  font-size: 0.76rem;
  font-weight: 500;
}
.tag-high {
  background: #FFE8D6;
  border-color: #F05A00;
  color: #F05A00;
  font-weight: 600;
}

.emotion-label {
  font-size: 0.78rem;
  color: #374151;
  margin-bottom: 0.22rem;
  display: flex;
  justify-content: space-between;
}
.emotion-bar-bg {
  background: #F3F0EC;
  border-radius: 100px;
  height: 6px;
  margin-bottom: 0.55rem;
  overflow: hidden;
}
.emotion-bar-fill {
  background: linear-gradient(90deg, #F05A00, #FF9640);
  height: 100%;
  border-radius: 100px;
}

.sim-badge {
  background: linear-gradient(90deg, #F05A00, #FF9640);
  color: white;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 0.15rem 0.55rem;
  border-radius: 100px;
  display: inline-block;
  margin-top: 0.35rem;
}
.retrieval-meta { font-size: 0.7rem; color: #9CA3AF; margin-top: 0.1rem; }

.info-banner {
  background: #FFF8F3;
  border: 1px solid #FFDEC7;
  border-left: 3px solid #F05A00;
  border-radius: 8px;
  padding: 0.8rem 1.1rem;
  font-size: 0.85rem;
  color: #7A3800;
  margin: 0.8rem 0;
}
.success-banner {
  background: #F0FDF4;
  border: 1px solid #BBF7D0;
  border-left: 3px solid #16A34A;
  border-radius: 8px;
  padding: 0.8rem 1.1rem;
  font-size: 0.85rem;
  color: #14532D;
  margin: 0.8rem 0;
}
.error-banner {
  background: #FEF2F2;
  border: 1px solid #FECACA;
  border-left: 3px solid #DC2626;
  border-radius: 8px;
  padding: 0.8rem 1.1rem;
  font-size: 0.85rem;
  color: #7F1D1D;
  margin: 0.8rem 0;
}

.embedding-preview {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  color: #6B7280;
  background: #FAFAFA;
  border: 1px solid #EEEBE6;
  border-radius: 8px;
  padding: 0.8rem;
  word-break: break-all;
  line-height: 1.6;
}

[data-baseweb="tab"] {
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: #F05A00 !important;
  border-bottom-color: #F05A00 !important;
}

/* Analysis page: default button */
.stButton > button {
  background: linear-gradient(135deg, #F05A00, #FF7A1A) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.55rem 1.5rem !important;
  box-shadow: 0 3px 12px rgba(240,90,0,0.25) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  box-shadow: 0 5px 18px rgba(240,90,0,0.38) !important;
  transform: translateY(-1px) !important;
}

.back-btn .stButton > button {
  background: transparent !important;
  color: #9A9A9A !important;
  border: 1px solid #E8E2DA !important;
  border-radius: 7px !important;
  box-shadow: none !important;
  font-size: 0.82rem !important;
  font-weight: 400 !important;
  padding: 0.38rem 0.9rem !important;
}
.back-btn .stButton > button:hover {
  background: #FAFAFA !important;
  color: #1C1C1C !important;
  box-shadow: none !important;
  transform: none !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
  background: #F05A00 !important;
}
hr { border-color: #F0EDE8 !important; margin: 1.5rem 0 !important; }

.fiq-footer {
  text-align: center;
  padding: 2rem;
  font-size: 0.78rem;
  color: #B0A8A0;
  border-top: 1px solid #F0EDE8;
  margin-top: 2rem;
}

.analysis-crumb {
  font-size: 0.8rem;
  color: #B0A8A0;
  padding: 0.8rem 0 0.2rem;
  display: flex;
  align-items: center;
  gap: 0.35rem;
}
.analysis-crumb b { color: #F05A00; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def uploaded_to_bgr(uploaded):
    pil = Image.open(uploaded).convert("RGB")
    return pil_to_bgr(pil)

def render_emotion_bars(probs, top_n=5):
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    html = ""
    for emotion, pct in sorted_probs:
        html += f"""
        <div class="emotion-label">
          <span>{emotion}</span>
          <span style="font-weight:600;color:#F05A00">{pct:.1f}%</span>
        </div>
        <div class="emotion-bar-bg">
          <div class="emotion-bar-fill" style="width:{pct}%"></div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_attribute_tags(attrs, threshold=0.55):
    html = '<div class="tag-wrap">'
    for attr, conf in sorted(attrs.items(), key=lambda x: x[1], reverse=True):
        if conf >= threshold:
            cls = "tag tag-high" if conf >= 0.75 else "tag"
            html += f'<span class="{cls}">{attr.replace("_"," ")} {int(conf*100)}%</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_retrieval_grid(results):
    if not results:
        st.markdown("""<div class="info-banner">📭 No similar faces found. Populate the
        database first with <code>python utils/demo_database.py</code></div>""",
        unsafe_allow_html=True)
        return
    cols = st.columns(min(len(results), 5))
    for col, item in zip(cols, results):
        with col:
            rank      = item.get("rank", "?")
            sim       = item.get("similarity_pct", 0)
            gender    = item.get("gender", "—")
            age       = item.get("age", "—")
            emotion   = item.get("emotion", "—")
            thumb     = item.get("thumbnail", "")
            loaded    = False
            if thumb and Path(thumb).exists():
                try:
                    col.image(str(thumb), use_container_width=True)
                    loaded = True
                except Exception:
                    pass
            if not loaded:
                ph = np.full((112, 112, 3), [255, 242, 230], dtype=np.uint8)
                cv2.putText(ph, f"#{rank}", (28, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 90, 0), 2)
                col.image(bgr_to_pil(ph), use_container_width=True)
            col.markdown(f"""
            <div style="text-align:center;padding:.25rem 0">
              <span class="sim-badge">{sim:.1f}% match</span>
              <div class="retrieval-meta">#{rank} · {gender} · {age}y</div>
              <div class="retrieval-meta">{emotion}</div>
            </div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from pipeline import FacePipeline
    return FacePipeline(device="cpu")

def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "home"


# ══════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════

def render_home():

    # Topbar
    st.markdown("""
    <div class="hp-topbar">
      <div class="hp-brand">Face<span>IQ</span></div>
      <div class="hp-nav-links">
        <span class="hp-nav-link">Semantic ImageRetrieval </span>
        <span class="hp-nav-link">Facial Characterization</span>        
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero section (pure HTML — no buttons yet)
    st.markdown("""
    <div class="hp-hero">
      <div class="hp-hero-left">
        <div class="hp-eyebrow">
          <span class="hp-eyebrow-line"></span>
          Computer Vision &nbsp;·&nbsp; Explainable AI
        </div>
        <h1 class="hp-headline">
          See beyond<br>what a face <em>reveals.</em>
        </h1>
        <p class="hp-desc">
          FaceIQ runs a six-stage deep learning pipeline on any face photo —
          detecting, characterising, and retrieving similar faces.
        </p>
        <div class="hp-pills">
          <span class="hp-pill">🔍 Face Detection</span>
          <span class="hp-pill">📐 40+ Attributes</span>
          <span class="hp-pill">😊 Emotion Recognition</span>
          <span class="hp-pill">🔮 Face Embeddings</span>
          <span class="hp-pill">🗃️ FAISS Retrieval</span>
          <span class="hp-pill">🌡️ Grad-CAM XAI</span>
        </div>
      </div>
      <div class="hp-hero-right">
        <div class="hp-stats-panel">
          <div class="hp-stats-title">Built for <em>precision.</em></div>
          <div class="hp-stats-sub">
            Every number comes from a distinct model stage,<br>combined into one coherent analysis.
          </div>
          <div class="hp-stat-grid">
            <div class="hp-stat-cell">
              <span class="hp-stat-num">40+</span>
              <span class="hp-stat-lbl">Face attributes</span>
            </div>
            <div class="hp-stat-cell">
              <span class="hp-stat-num">512</span>
              <span class="hp-stat-lbl">Embedding dims</span>
            </div>
            <div class="hp-stat-cell">
              <span class="hp-stat-num">7</span>
              <span class="hp-stat-lbl">Emotion classes</span>
            </div>
            <div class="hp-stat-cell">
              <span class="hp-stat-num">~1s</span>
              <span class="hp-stat-lbl">Inference time</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero CTA buttons (Streamlit-managed)
    _, c1,_r = st.columns([1.2, 1.1, 4.2])
    with c1:
        st.markdown('<div class="hp-cta-btn">', unsafe_allow_html=True)
        if st.button("Try Analysis  →", use_container_width=True, key="hero_cta"):
            st.session_state.page = "analysis"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How it works strip
    st.markdown("""
    <div class="hp-how-strip">
      <div class="hp-how-cell">
        <div class="hp-how-num">01</div>
        <div class="hp-how-title">Upload a photo</div>
        <div class="hp-how-desc">Drop any JPEG or PNG, or snap directly from your webcam.</div>
      </div>
      <div class="hp-how-cell">
        <div class="hp-how-num">02</div>
        <div class="hp-how-title">Detect &amp; align</div>
        <div class="hp-how-desc">Faster R-CNN locates the face and crops it to a normalised region.</div>
      </div>
      <div class="hp-how-cell">
        <div class="hp-how-num">03</div>
        <div class="hp-how-title">Predict &amp; embed</div>
        <div class="hp-how-desc">Age, gender, 40+ attributes, emotion probabilities and a 512-dim embedding are computed.</div>
      </div>
      <div class="hp-how-cell">
        <div class="hp-how-num">04</div>
        <div class="hp-how-title">Retrieve &amp; explain</div>
        <div class="hp-how-desc">FAISS finds similar faces; Grad-CAM highlights why the model made each decision.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline architecture cards
    st.markdown("""
    <div class="hp-section">
      <div class="hp-section-header">
        <span class="hp-section-num">— Architecture</span>
        <span class="hp-section-title">Six-stage pipeline</span>
        <span class="hp-section-rule"></span>
      </div>
      <div class="hp-pipeline-grid">
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 01</div>
          <span class="hp-pipe-icon">🔍</span>
          <div class="hp-pipe-name">Face Detection</div>
          <div class="hp-pipe-desc">Faster R-CNN locates faces with bounding box confidence. Handles multiple faces gracefully.</div>
        </div>
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 02</div>
          <span class="hp-pipe-icon">📐</span>
          <div class="hp-pipe-name">Attribute Prediction</div>
          <div class="hp-pipe-desc">ResNet-50 backbone predicts over 40 CelebA attributes — hair colour, accessories, expression cues.</div>
        </div>
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 03</div>
          <span class="hp-pipe-icon">😊</span>
          <div class="hp-pipe-name">Emotion Recognition</div>
          <div class="hp-pipe-desc">MobileNetV2 classifies 7 emotional states with full softmax probability distribution.</div>
        </div>
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 04</div>
          <span class="hp-pipe-icon">🔮</span>
          <div class="hp-pipe-name">Face Embedding</div>
          <div class="hp-pipe-desc">ArcFace produces a discriminative 512-dim L2-normalised vector for identity representation.</div>
        </div>
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 05</div>
          <span class="hp-pipe-icon">🗃️</span>
          <div class="hp-pipe-name">Similarity Retrieval</div>
          <div class="hp-pipe-desc">Two-stage retrieval: attribute pre-filter then FAISS cosine re-ranking for top-K results.</div>
        </div>
        <div class="hp-pipe-card">
          <div class="hp-pipe-step">Stage 06</div>
          <span class="hp-pipe-icon">🌡️</span>
          <div class="hp-pipe-name">Grad-CAM XAI</div>
          <div class="hp-pipe-desc">Gradient-weighted activation maps reveal which facial regions drove each prediction.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Bottom CTA bar (dark strip)
    st.markdown("""
    <div class="hp-cta-bar">
      <div class="hp-cta-left">
        Ready to analyse a face?<br><span>Upload one now.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA button inside dark strip (positioned via negative margin)
    _, cta_c, _r2 = st.columns([5.2, 1.5, 4.8])
    with cta_c:
        st.markdown('<div class="hp-cta-btn-white" style="margin-top:-3.8rem;position:relative;z-index:10">',
                    unsafe_allow_html=True)
        if st.button("Open Analysis  →", use_container_width=True, key="cta_bottom"):
            st.session_state.page = "analysis"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="hp-footer">
      <div class="hp-footer-left">© 2026 FaceIQ · PyTorch · Streamlit · FAISS</div>
      <div class="hp-footer-right">
        <span>Research</span>
        <span>Privacy</span>
        <span>GitHub</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  ANALYSIS PAGE
# ══════════════════════════════════════════════════

def render_analysis():

    # Breadcrumb + back button
    col_crumb, col_back = st.columns([7, 1])
    with col_crumb:
        st.markdown("""
        <div class="analysis-crumb">
          FaceIQ &nbsp;›&nbsp; <b>Analysis</b>
        </div>""", unsafe_allow_html=True)
    with col_back:
        st.markdown('<div class="back-btn" style="padding-top:0.6rem">', unsafe_allow_html=True)
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="fiq-header">
      <p class="fiq-logo">Face<span>IQ</span><span class="fiq-badge">Beta</span></p>
      <p class="fiq-subtitle">
        Upload or capture a face · Intelligent attribute prediction · Explainable similarity retrieval
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        top_k              = st.slider("Top-K Results", 1, 10, 5)
        enable_attr_filter = st.toggle("Enable Attribute Filtering", value=True)
        generate_gradcam   = st.toggle("Generate Grad-CAM", value=True)
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.8rem;color:#9CA3AF;line-height:1.7">
          <b>Pipeline</b><br>
          🔍 Faster R-CNN Detection<br>
          📐 ResNet Attributes<br>
          😊 MobileNetV2 Emotion<br>
          🔮 ArcFace Embeddings<br>
          🗃️ FAISS Retrieval<br>
          🌡️ Grad-CAM XAI
        </div>""", unsafe_allow_html=True)

    # Input card
    st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Input</div>', unsafe_allow_html=True)
    input_tab, webcam_tab = st.tabs(["📁  Upload Image", "📷  Webcam"])
    input_image_bgr = None

    with input_tab:
        uploaded = st.file_uploader("Drag & drop a face image",
                                    type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded:
            input_image_bgr = uploaded_to_bgr(uploaded)
            st.markdown('<div class="success-banner">✅ Image loaded successfully.</div>',
                        unsafe_allow_html=True)
    with webcam_tab:
        cam_frame = st.camera_input("", label_visibility="collapsed")
        if cam_frame:
            input_image_bgr = uploaded_to_bgr(cam_frame)
            st.markdown('<div class="success-banner">✅ Frame captured.</div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if input_image_bgr is None:
        st.markdown("""<div class="info-banner">
          👆 Upload a JPEG/PNG or use your webcam. Ensure the face is clearly visible.
        </div>""", unsafe_allow_html=True)
        _render_footer()
        return

    # Preview
    st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖼️ Preview</div>', unsafe_allow_html=True)
    p1, _ = st.columns(2)
    with p1:
        st.caption("Input Image")
        h = int(400 * input_image_bgr.shape[0] / input_image_bgr.shape[1])
        st.image(bgr_to_pil(cv2.resize(input_image_bgr, (400, h))), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyse button
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run = st.button("🔍  Analyze Face", use_container_width=True)
    if not run:
        _render_footer()
        return

    # Processing
    with st.spinner("Analyzing face… This may take a moment on first run."):
        t0 = time.time()
        pipeline = get_pipeline()
        result   = pipeline.process(
            image_bgr=input_image_bgr,
            top_k=top_k,
            enable_attribute_filter=enable_attr_filter,
            generate_gradcam=generate_gradcam,
        )
        elapsed = time.time() - t0

    if not result.success or result.error:
        st.markdown(f'<div class="error-banner">❌ <b>Error:</b> {result.error or "Unknown error."}</div>',
                    unsafe_allow_html=True)
        _render_footer()
        return

    st.markdown(f"""<div class="success-banner">
      ✅ <b>Face processed</b> — {elapsed:.2f}s (backend: {result.embedding_backend})
    </div>""", unsafe_allow_html=True)

    # Detection result
    st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Detection Result</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Annotated Input")
        if result.annotated_image_bgr is not None:
            h = int(400 * result.annotated_image_bgr.shape[0] / result.annotated_image_bgr.shape[1])
            st.image(bgr_to_pil(cv2.resize(result.annotated_image_bgr, (400, h))),
                     use_container_width=True)
    with d2:
        st.caption("Detected & Aligned Face")
        if result.detected_face_bgr is not None:
            st.image(bgr_to_pil(cv2.resize(result.detected_face_bgr, (224, 224))), width=224)
        conf_pct = int(result.detection_confidence * 100)
        st.markdown(f"""
        <div style="margin-top:.8rem">
          <span style="font-size:.8rem;color:#9A9A9A">Detection Confidence</span><br>
          <span style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#F05A00">{conf_pct}%</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Age / Gender / Emotion
    st.markdown('<div class="fiq-card fiq-card-accent">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Predicted Attributes</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1, 2])
    with r1:
        g_sym = "♂" if result.gender == "Male" else "♀"
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-chip"><span class="val">{result.age}</span><span class="lbl">Age</span></div>
        </div>
        <div class="stat-row" style="margin-top:.5rem">
          <div class="stat-chip">
            <span class="val" style="font-size:1.2rem">{g_sym}</span>
            <span class="lbl">{result.gender}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with r2:
      emoji = result.emotion_emoji or "😐"

      st.markdown(f"""
      <div style="text-align:center;padding:.5rem">
        <div style="font-size:3.2rem;line-height:1">{emoji}</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:600;
                  color:#1C1C1C;margin-top:.4rem">{result.emotion}</div>
      </div>
      """, unsafe_allow_html=True)

      st.markdown("<div style='text-align:center;font-weight:600;margin-top:10px'>Emotion Breakdown</div>",
                unsafe_allow_html=True)

      if result.emotion_all_probs:
        render_emotion_bars(result.emotion_all_probs)
    st.markdown('</div>', unsafe_allow_html=True)

    # CelebA attributes
    st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ Face Attributes</div>', unsafe_allow_html=True)
    if result.prominent_attributes:
        render_attribute_tags(result.prominent_attributes)
    else:
        st.markdown('<div class="info-banner">No prominent attributes above threshold.</div>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**All attribute scores:**")
    if result.attributes:
        acols = st.columns(4)
        for i, (attr, conf) in enumerate(
                sorted(result.attributes.items(), key=lambda x: x[1], reverse=True)):
            with acols[i % 4]:
                color = "#F05A00" if conf >= 0.55 else "#C0B8B0"
                st.markdown(f"""
                <div style="margin-bottom:.5rem">
                  <div style="font-size:.76rem;color:#374151;margin-bottom:2px">{attr.replace("_"," ")}</div>
                  <div style="background:#F3F0EC;border-radius:100px;height:4px;overflow:hidden">
                    <div style="width:{int(conf*100)}%;background:{color};height:100%;border-radius:100px"></div>
                  </div>
                  <div style="font-size:.7rem;color:{color};font-weight:500">{int(conf*100)}%</div>
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Embedding
    if result.embedding is not None:
        st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Face Embedding</div>', unsafe_allow_html=True)
        emb = result.embedding
        st.markdown(f"""
        <div class="embedding-preview">
          dim={emb.shape[0]} · norm={np.linalg.norm(emb):.4f} · backend={result.embedding_backend}<br>
          [{", ".join(f"{v:.4f}" for v in emb[:20])} ...]
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Retrieval
    st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🗃️ Similar Faces · Top {top_k}</div>',
                unsafe_allow_html=True)
    if enable_attr_filter:
        st.markdown("""<div class="info-banner">
          🔎 <b>Stage 1:</b> Attribute filtering &nbsp;→&nbsp;
          <b>Stage 2:</b> Cosine similarity re-ranking (512-dim embeddings)
        </div>""", unsafe_allow_html=True)
    render_retrieval_grid(result.retrieved_faces)
    st.markdown('</div>', unsafe_allow_html=True)

    # Grad-CAM
    if generate_gradcam and (result.emotion_gradcam_bgr is not None
                             or result.attribute_gradcam_bgr is not None):
        st.markdown('<div class="fiq-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌡️ Grad-CAM Explainability</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-banner">
          Heatmaps show which facial regions most influenced each prediction.
          <b>Red = high attention · Blue = low attention</b>
        </div>""", unsafe_allow_html=True)
        x1, x2, x3 = st.columns(3)
        with x1:
            st.caption("Original Face")
            if result.detected_face_bgr is not None:
                st.image(bgr_to_pil(result.detected_face_bgr), use_container_width=True)
        with x2:
            if result.emotion_gradcam_bgr is not None:
                st.caption(f"Emotion: {result.emotion} ({result.emotion_confidence}%)")
                st.image(bgr_to_pil(result.emotion_gradcam_bgr), use_container_width=True)
        with x3:
            if result.attribute_gradcam_bgr is not None:
                best_attr = (max(result.prominent_attributes, key=result.prominent_attributes.get)
                             if result.prominent_attributes else "Attribute")
                conf = int(result.prominent_attributes.get(best_attr, 0) * 100)
                st.caption(f"Attribute: {best_attr.replace('_',' ')} ({conf}%)")
                st.image(bgr_to_pil(result.attribute_gradcam_bgr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    _render_footer()


def _render_footer():
    st.markdown("""
    <div class="fiq-footer">
      FaceIQ · Hybrid Facial Characterization & Explainable Retrieval ·
      Built with PyTorch · Streamlit · FAISS
    </div>
    """, unsafe_allow_html=True)


# ── Router ───────────────────────────────────────────────────────────────────

def main():
    init_state()
    if st.session_state.page == "home":
        render_home()
    else:
        render_analysis()

if __name__ == "__main__":
    main()
