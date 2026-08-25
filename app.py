import os
import html as html_lib
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# CONFIG
# =====================================================================
API_BASE = (os.environ.get("API_BASE") or "http://localhost:8000").rstrip("/")
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="CineMatch — AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# GOOGLE FONTS
# =====================================================================
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# =====================================================================
# PREMIUM DARK THEME — CSS INJECTION
# =====================================================================
st.markdown(
    """
<style>
/* ══════════════════════════════════════════════════════════
   CineMatch — Premium Dark Glassmorphism Theme
   ══════════════════════════════════════════════════════════ */

/* ── Root Variables ────────────────────────────────────── */
:root {
    --bg-deep:       #0a0a14;
    --bg-card:       rgba(18, 18, 40, 0.55);
    --bg-card-hover: rgba(28, 28, 60, 0.75);
    --bg-glass:      rgba(255,255,255,0.04);
    --bg-glass-h:    rgba(255,255,255,0.08);
    --accent:        #7c5cfc;
    --accent2:       #00d4ff;
    --gold:          #fbbf24;
    --txt:           #eeeef5;
    --txt2:          #9898b0;
    --txt3:          #6b6b88;
    --bdr:           rgba(255,255,255,0.06);
    --bdr2:          rgba(255,255,255,0.10);
    --bdr-accent:    rgba(124,92,252,0.3);
    --radius:        12px;
    --radius-lg:     16px;
    --radius-xl:     24px;
    --shadow-card:   0 4px 24px rgba(0,0,0,0.3);
    --shadow-glow:   0 0 28px rgba(124,92,252,0.15);
    --tr:            0.3s cubic-bezier(.4,0,.2,1);
}

/* ── Global Background ─────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 18% 8%, rgba(124,92,252,0.07) 0%, transparent 50%),
        radial-gradient(ellipse 60% 50% at 82% 85%, rgba(0,212,255,0.05) 0%, transparent 50%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stApp"], .main, .main .block-container {
    background: transparent !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* ── Header / Toolbar ──────────────────────────────────── */
[data-testid="stHeader"] {
    background: rgba(10,10,20,0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid var(--bdr) !important;
}
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(12,12,28,0.97) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--bdr) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    color: var(--txt) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Typography ────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--txt) !important;
    letter-spacing: -0.3px !important;
}
h1 { font-weight: 800 !important; }
p, span, label, div { color: var(--txt) !important; }

/* ── Text Input (Search) ───────────────────────────────── */
[data-testid="stTextInput"] > div > div > input {
    background: var(--bg-glass) !important;
    border: 1px solid var(--bdr2) !important;
    border-radius: var(--radius-xl) !important;
    color: var(--txt) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 20px !important;
    transition: all var(--tr) !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    background: var(--bg-glass-h) !important;
    border-color: var(--bdr-accent) !important;
    box-shadow: 0 0 24px rgba(124,92,252,0.18) !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: var(--txt3) !important;
}
[data-testid="stTextInput"] > label {
    color: var(--txt2) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Selectbox ─────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--bdr2) !important;
    border-radius: var(--radius) !important;
    color: var(--txt) !important;
}
[data-testid="stSelectbox"] label {
    color: var(--txt2) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Buttons ───────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(124,92,252,0.12), rgba(0,212,255,0.12)) !important;
    border: 1px solid var(--bdr-accent) !important;
    color: #c4c4e0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    transition: all var(--tr) !important;
    letter-spacing: 0.2px !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(124,92,252,0.28), rgba(0,212,255,0.28)) !important;
    border-color: rgba(124,92,252,0.55) !important;
    color: #ffffff !important;
    box-shadow: var(--shadow-glow) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Sidebar home button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-xl) !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 0 30px rgba(124,92,252,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Divider ───────────────────────────────────────────── */
[data-testid="stHorizontalRule"], hr {
    border-color: var(--bdr) !important;
    opacity: 0.5 !important;
}

/* ── Images ────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    transition: transform var(--tr), box-shadow var(--tr) !important;
}
[data-testid="stImage"]:hover img {
    transform: scale(1.03) !important;
    box-shadow: var(--shadow-card) !important;
}

/* ── Alerts / Info / Error ─────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--bdr2) !important;
    border-radius: var(--radius) !important;
    color: var(--txt2) !important;
}

/* ── Expander ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: var(--radius) !important;
}

/* ── Custom Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb {
    background: rgba(124,92,252,0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(124,92,252,0.45); }

/* ══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ══════════════════════════════════════════════════════════ */

/* ── Movie Card ────────────────────────────────────────── */
.mc-card {
    background: var(--bg-card);
    border: 1px solid var(--bdr);
    border-radius: var(--radius);
    overflow: hidden;
    transition: all var(--tr);
    margin-bottom: 4px;
}
.mc-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--bdr-accent);
    box-shadow: var(--shadow-glow), var(--shadow-card);
    transform: translateY(-6px);
}
.mc-poster-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2/3;
    overflow: hidden;
    background: #111128;
}
.mc-poster {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform var(--tr);
}
.mc-card:hover .mc-poster {
    transform: scale(1.06);
}
.mc-no-poster {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    background: linear-gradient(135deg, #111128, #1a1a3a);
}
.mc-rating {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(0,0,0,0.72);
    backdrop-filter: blur(8px);
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--gold);
}
.mc-info {
    padding: 10px 10px 8px;
}
.mc-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--txt) !important;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.1em;
}
.mc-year {
    font-size: 0.72rem;
    color: var(--txt3) !important;
    margin-top: 3px;
}

/* ── Hero Banner ───────────────────────────────────────── */
.hero-banner {
    position: relative;
    text-align: center;
    padding: 48px 20px 36px;
    margin: -2rem -1rem 1.5rem;
    overflow: hidden;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(124,92,252,0.1), rgba(0,212,255,0.06));
    z-index: 0;
}
.hero-title {
    position: relative;
    z-index: 1;
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: var(--txt) !important;
    margin-bottom: 8px !important;
    letter-spacing: -1px !important;
    line-height: 1.2 !important;
}
.hero-gradient {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    position: relative;
    z-index: 1;
    font-size: 1rem !important;
    color: var(--txt2) !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}

/* ── Category Tabs ─────────────────────────────────────── */
.cat-tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.cat-tab {
    padding: 7px 14px;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all var(--tr);
    background: var(--bg-glass);
    color: var(--txt2);
    text-decoration: none;
    display: block;
    width: 100%;
    box-sizing: border-box;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.2px;
}
.cat-tab:hover {
    background: var(--bg-glass-h);
    color: var(--txt);
}
.cat-tab.active {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    border-color: transparent;
    box-shadow: var(--shadow-glow);
}

/* ── Section Titles ────────────────────────────────────── */
.sec-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--txt) !important;
    margin-bottom: 20px !important;
    letter-spacing: -0.3px !important;
}
.sec-badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    vertical-align: middle;
    margin-left: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Detail Backdrop Hero ──────────────────────────────── */
.detail-backdrop {
    position: relative;
    width: calc(100% + 2rem);
    margin: -2rem -1rem 0;
    height: 360px;
    overflow: hidden;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}
.detail-backdrop img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.35) saturate(1.2);
}
.detail-backdrop .db-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(10,10,20,0.3) 0%, rgba(10,10,20,0.8) 60%, var(--bg-deep) 100%);
}
.detail-backdrop .db-title {
    position: absolute;
    bottom: 28px;
    left: 32px;
    right: 32px;
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1.15;
    text-shadow: 0 2px 12px rgba(0,0,0,0.5);
}

/* ── Detail Info Card ──────────────────────────────────── */
.detail-info-card {
    background: var(--bg-card);
    border: 1px solid var(--bdr);
    border-radius: var(--radius-lg);
    padding: 24px;
    backdrop-filter: blur(12px);
}
.detail-genres {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 16px;
}
.genre-pill {
    padding: 5px 14px;
    border-radius: var(--radius-xl);
    background: var(--bg-glass);
    border: 1px solid var(--bdr2);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--txt2);
    transition: all var(--tr);
}
.genre-pill:hover {
    background: var(--bg-glass-h);
    border-color: var(--bdr-accent);
    color: var(--txt);
}
.detail-meta-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 18px;
    font-size: 0.88rem;
    color: var(--txt2);
}
.detail-overview {
    font-size: 0.95rem;
    line-height: 1.8;
    color: var(--txt2) !important;
    margin-top: 8px;
}

/* ── Empty State ───────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 50px 20px;
    color: var(--txt3);
}
.empty-icon { font-size: 2.8rem; margin-bottom: 10px; }
.empty-text { font-size: 0.95rem; color: var(--txt2); }

/* ── Footer ────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 32px 20px;
    margin-top: 40px;
    border-top: 1px solid var(--bdr);
}
.footer-txt { font-size: 0.85rem; color: var(--txt3) !important; margin-bottom: 4px; }
.footer-sub { font-size: 0.75rem; color: var(--txt3) !important; }
.footer-sub a { color: var(--accent2); text-decoration: none; }

/* ── Animations ────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.mc-card { animation: fadeIn 0.45s ease both; }
.mc-card:nth-child(1)  { animation-delay: 0.02s; }
.mc-card:nth-child(2)  { animation-delay: 0.04s; }
.mc-card:nth-child(3)  { animation-delay: 0.06s; }

/* ══════════════════════════════════════════════════════════
   RESPONSIVE DESIGN
   ══════════════════════════════════════════════════════════ */

/* Make Streamlit columns wrap on smaller viewports */
@media (max-width: 900px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 0 0 calc(33.333% - 0.5rem) !important;
        min-width: 0 !important;
    }
    .hero-title { font-size: 1.8rem !important; }
    .hero-sub { font-size: 0.9rem !important; }
    .detail-backdrop { height: 260px; }
    .detail-backdrop .db-title { font-size: 1.6rem; bottom: 20px; left: 20px; right: 20px; }
}

@media (max-width: 600px) {
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 0 0 calc(50% - 0.5rem) !important;
    }
    .block-container { padding: 1rem 0.8rem !important; }
    .hero-banner { padding: 32px 12px 24px; }
    .hero-title { font-size: 1.5rem !important; }
    .detail-backdrop { height: 200px; }
    .detail-backdrop .db-title { font-size: 1.3rem; bottom: 14px; left: 14px; right: 14px; }
    .mc-info { padding: 7px 7px 5px; }
    .mc-title { font-size: 0.75rem; }
    .mc-year { font-size: 0.65rem; }
    .mc-rating { font-size: 0.62rem; padding: 2px 6px; }
    .sec-title { font-size: 1.15rem !important; }
    .cat-tab { padding: 6px 12px; font-size: 0.78rem; }
}

@media (max-width: 400px) {
    .hero-title { font-size: 1.3rem !important; }
    .mc-info { padding: 5px 5px 4px; }
    .mc-title { font-size: 0.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# STATE + ROUTING
# =====================================================================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None
if "home_cat" not in st.session_state:
    st.session_state.home_cat = "trending"

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =====================================================================
# API HELPERS
# =====================================================================
@st.cache_data(ttl=60)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


# =====================================================================
# POSTER GRID — Premium Cards
# =====================================================================
def poster_grid(cards, cols=6, key_prefix="grid"):
    """Render a responsive grid of movie poster cards."""
    if not cards:
        st.markdown(
            """
        <div class="empty-state">
            <div class="empty-icon">🎬</div>
            <div class="empty-text">No movies to show</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        columns = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = html_lib.escape(m.get("title", "Untitled"))
            poster = m.get("poster_url")
            year = (m.get("release_date") or "")[:4]
            rating = m.get("vote_average")

            with columns[c]:
                # Rating badge HTML
                rating_html = ""
                if rating and float(rating) > 0:
                    rating_html = (
                        f'<div class="mc-rating">⭐ {float(rating):.1f}</div>'
                    )

                # Poster HTML
                if poster:
                    poster_html = f'<img src="{poster}" class="mc-poster" loading="lazy" alt="{title}">'
                else:
                    poster_html = '<div class="mc-no-poster">🎬</div>'

                st.markdown(
                    f"""
                <div class="mc-card">
                    <div class="mc-poster-wrap">
                        {poster_html}
                        {rating_html}
                    </div>
                    <div class="mc-info">
                        <div class="mc-title">{title}</div>
                        <div class="mc-year">{year}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if tmdb_id:
                    if st.button(
                        "View",
                        key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}",
                        width="stretch",
                    ):
                        goto_details(tmdb_id)


# =====================================================================
# DATA HELPERS
# =====================================================================
def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                    "vote_average": tmdb.get("vote_average"),
                    "release_date": tmdb.get("release_date"),
                }
            )
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    """Parse TMDB search results into suggestions and cards."""
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                }
            )
    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                    "vote_average": m.get("vote_average"),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {
            "tmdb_id": x["tmdb_id"],
            "title": x["title"],
            "poster_url": x["poster_url"],
            "vote_average": x.get("vote_average"),
            "release_date": x.get("release_date"),
        }
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =====================================================================
# SIDEBAR
# =====================================================================
CATEGORIES = {
    "trending": "🔥 Trending",
    "popular": "⭐ Popular",
    "top_rated": "🏆 Top Rated",
    "now_playing": "🎥 Now Playing",
    "upcoming": "📅 Upcoming",
}

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown(
        "<p style='color:#6b6b88 !important; font-size:0.8rem; margin-top:-8px;'>AI Movie Recommendations</p>",
        unsafe_allow_html=True,
    )
    if st.button("🏠  Home", width="stretch"):
        goto_home()

    st.markdown("---")
    st.markdown("### Category")
    home_category = st.selectbox(
        "Select category",
        list(CATEGORIES.keys()),
        format_func=lambda x: CATEGORIES[x],
        index=list(CATEGORIES.keys()).index(st.session_state.home_cat),
        label_visibility="collapsed",
    )
    st.session_state.home_cat = home_category


# =====================================================================
# VIEW: HOME
# =====================================================================
if st.session_state.view == "home":

    # ── Hero Banner ──
    st.markdown(
        """
    <div class="hero-banner">
        <div class="hero-title">
            Discover Your Next <span class="hero-gradient">Favorite Movie</span>
        </div>
        <p class="hero-sub">AI-powered recommendations from thousands of films. Search, explore, and find hidden gems.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Search ──
    typed = st.text_input(
        "🔍 Search movies",
        placeholder="Try: Inception, Batman, Interstellar...",
        label_visibility="collapsed",
    )

    # ── Category Tabs (functional buttons) ──
    cat_cols = st.columns(len(CATEGORIES))
    for i, (key, label) in enumerate(CATEGORIES.items()):
        with cat_cols[i]:
            is_active = key == st.session_state.home_cat
            if is_active:
                st.markdown(
                    f'<div class="cat-tab active" style="text-align:center;">{label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(label, key=f"cat_tab_{key}", use_container_width=True):
                    st.session_state.home_cat = key
                    st.rerun()

    st.markdown("---")

    # ── SEARCH MODE ──
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            with st.spinner("Searching..."):
                data, err = api_get_json(
                    "/tmdb/search", params={"query": typed.strip()}
                )

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                if suggestions:
                    labels = ["— Select a movie —"] + [s[0] for s in suggestions]
                    selected = st.selectbox("💡 Quick pick", labels, index=0)

                    if selected != "— Select a movie —":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown(
                    '<div class="sec-title">🔍 Search Results</div>',
                    unsafe_allow_html=True,
                )
                poster_grid(cards, cols=6, key_prefix="search_results")

        st.stop()

    # ── HOME FEED ──
    active_category = st.session_state.home_cat
    cat_display = CATEGORIES.get(active_category, active_category)
    st.markdown(
        f'<div class="sec-title">{cat_display}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading movies..."):
        home_cards, err = api_get_json(
            "/home", params={"category": active_category, "limit": 24}
        )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=6, key_prefix="home_feed")

    # ── Footer ──
    st.markdown(
        """
    <div class="app-footer">
        <p class="footer-txt">Built using FastAPI + TMDB • AI-Powered by TF-IDF</p>
        <p class="footer-sub">© 2026 CineMatch · Data by <a href="https://www.themoviedb.org/" target="_blank">TMDB</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =====================================================================
# VIEW: DETAILS
# =====================================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # ── Back Button ──
    if st.button("← Back to Home"):
        goto_home()

    # ── Fetch Details ──
    with st.spinner("Loading movie details..."):
        data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    title_safe = html_lib.escape(data.get("title", ""))
    overview = html_lib.escape(data.get("overview") or "No overview available.")
    release = data.get("release_date") or "—"
    genres = data.get("genres") or []
    genre_names = ", ".join([g["name"] for g in genres]) or "—"
    poster_url = data.get("poster_url")
    backdrop_url = data.get("backdrop_url")

    # ── Backdrop Hero ──
    if backdrop_url:
        st.markdown(
            f"""
        <div class="detail-backdrop">
            <img src="{backdrop_url}" alt="Backdrop">
            <div class="db-overlay"></div>
            <div class="db-title">{title_safe}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h1 style='font-family:Outfit,sans-serif; font-weight:800; letter-spacing:-0.5px;'>{title_safe}</h1>",
            unsafe_allow_html=True,
        )

    # ── Poster + Info Layout ──
    col_poster, col_info = st.columns([1, 2.5], gap="large")

    with col_poster:
        if poster_url:
            st.image(poster_url, width="stretch")
        else:
            st.markdown(
                '<div style="text-align:center; padding:40px; font-size:4rem; background:rgba(18,18,40,0.5); border-radius:12px;">🎬</div>',
                unsafe_allow_html=True,
            )

    with col_info:
        if backdrop_url:
            # Title already shown in backdrop
            pass
        st.markdown(
            f"<h2 style='font-family:Outfit,sans-serif; font-weight:700; margin-bottom:6px;'>{title_safe}</h2>",
            unsafe_allow_html=True,
        )

        # Meta row
        st.markdown(
            f"""
        <div style="margin-bottom:8px;">
            <span class="detail-meta-item">📅 {release}</span>
            <span class="detail-meta-item">🎭 {genre_names}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Genre pills
        if genres:
            pills_html = '<div class="detail-genres">'
            for g in genres:
                pills_html += f'<span class="genre-pill">{html_lib.escape(g["name"])}</span>'
            pills_html += "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

        # Overview
        st.markdown("#### Overview")
        st.markdown(
            f'<p class="detail-overview">{overview}</p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Recommendations ──
    st.markdown(
        '<div class="sec-title">✅ Recommendations</div>',
        unsafe_allow_html=True,
    )

    title = (data.get("title") or "").strip()
    if title:
        with st.spinner("Finding recommendations..."):
            bundle, err2 = api_get_json(
                "/movie/search",
                params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
            )

        if not err2 and bundle:
            st.markdown(
                '<div class="sec-title">🔎 Similar Movies <span class="sec-badge">TF-IDF AI</span></div>',
                unsafe_allow_html=True,
            )
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=6,
                key_prefix="details_tfidf",
            )

            st.markdown(
                '<div class="sec-title">🎭 More Like This <span class="sec-badge">Genre</span></div>',
                unsafe_allow_html=True,
            )
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=6,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing genre recommendations (fallback).")
            with st.spinner("Loading genre recommendations..."):
                genre_only, err3 = api_get_json(
                    "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
                )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=6, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")

    # ── Footer ──
    st.markdown(
        """
    <div class="app-footer">
        <p class="footer-txt">Built using FastAPI + TMDB • AI-Powered by TF-IDF</p>
        <p class="footer-sub">© 2026 CineMatch · Data by <a href="https://www.themoviedb.org/" target="_blank">TMDB</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
