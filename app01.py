import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =========================
# 0) Page config
# =========================
st.set_page_config(
    page_title="이탈 위험 모니터링",
    page_icon="📉",
    layout="wide",
)

# =========================
# 1) Global CSS (LIGHT THEME)
# =========================
def inject_css_light():
    st.markdown(
        """
        <style>
        /* =========================
           Global tokens (Light)
        ========================= */
        :root {
            --bg: #ffffff;
            --surface: #ffffff;
            --surface-2: #f8fafc;
            --border: rgba(15,23,42,0.10);
            --border-2: rgba(15,23,42,0.08);

            --text: #0f172a;          /* slate-900 */
            --text-strong: #0b1220;
            --text-muted: rgba(15,23,42,0.68);

            --primary: #2563eb;       /* blue-600 */
            --danger: #ef4444;        /* red-500 */
            --warning: #f59e0b;       /* amber-500 */
            --success: #16a34a;       /* green-600 */

            --shadow: 0 10px 22px rgba(15,23,42,0.06);
            --radius-lg: 18px;
            --radius-md: 16px;
            --radius-sm: 12px;
        }

        /* App background */
        .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        /* Container padding */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* Ensure all text defaults to dark (fix black-on-black from some themes) */
        html, body, [class*="st-"], p, span, div, label {
            color: var(--text) !important;
        }

        /* Titles */
        .big-title {
            font-size: 54px;
            font-weight: 900;
            margin-bottom: 4px;
            color: var(--text-strong) !important;
            letter-spacing: -1px;
        }
        .subtitle {
            color: var(--text-muted) !important;
            font-size: 16px;
            margin-bottom: 18px;
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.4px;
            color: var(--text-strong) !important;
        }

        /* =========================
           Sidebar
        ========================= */
        section[data-testid="stSidebar"] {
            background: var(--surface-2) !important;
            border-right: 1px solid var(--border-2) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        /* =========================
           Panels / Cards
        ========================= */
        .panel {
            border: 1px solid var(--border);
            background: var(--surface) !important;
            border-radius: var(--radius-lg);
            padding: 18px;
            box-shadow: var(--shadow);
        }

        .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 8px; }
        .kpi-card {
            border: 1px solid var(--border);
            background: var(--surface) !important;
            border-radius: var(--radius-md);
            padding: 18px 18px 14px 18px;
            box-shadow: var(--shadow);
            min-height: 110px;
        }
        .kpi-title { font-size: 14px; color: var(--text-muted) !important; margin-bottom: 10px; }
        .kpi-value { font-size: 44px; font-weight: 900; line-height: 1.0; color: var(--text-strong) !important; }
        .kpi-sub { margin-top: 10px; font-size: 12px; color: rgba(15,23,42,0.55) !important; }

        .cust-card {
            border: 1px solid var(--border);
            background: var(--surface) !important;
            border-radius: var(--radius-md);
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 8px 18px rgba(15,23,42,0.05);
        }
        .cust-grid {
            display: grid;
            grid-template-columns: 1.3fr 1.8fr 1fr 0.9fr;
            gap: 10px;
            align-items: center;
        }

        .muted { color: var(--text-muted) !important; font-size: 13px; }

        /* Tag - make contrast clear */
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 10px;
            background: rgba(22, 163, 74, 0.10) !important;
            color: #166534 !important; /* green-800 */
            border: 1px solid rgba(22, 163, 74, 0.22) !important;
            font-weight: 800;
            font-size: 12px;
            margin-left: 8px;
        }

        /* =========================
           Streamlit widgets (critical)
           Fix: some widgets inherit dark theme styles and become black-on-black.
        ========================= */

        /* Inputs (selectbox, text_input, multiselect, etc.) */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        /* Selectbox / Multiselect */
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }
        div[data-baseweb="select"] span {
            color: var(--text) !important;
        }

        /* Dropdown menu */
        ul[role="listbox"] {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }
        ul[role="listbox"] * {
            color: var(--text) !important;
        }

        /* Slider */
        div[data-testid="stSlider"] * {
            color: var(--text) !important;
        }

        /* Buttons */
        button[kind="secondary"], button[kind="primary"] {
            border-radius: 12px !important;
        }
        /* Make secondary button readable on white */
        button[kind="secondary"] {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }
        /* Primary button readable */
        button[kind="primary"] {
            background: var(--primary) !important;
            color: #ffffff !important;
            border: 1px solid rgba(37,99,235,0.25) !important;
        }

        /* Metric component */
        div[data-testid="stMetric"] {
            background: transparent !important;
        }
        div[data-testid="stMetric"] * {
            color: var(--text) !important;
        }
        div[data-testid="stMetric"] label {
            color: var(--text-muted) !important;
        }

        /* Dataframe/table */
        .stDataFrame, .stTable {
            background: #ffffff !important;
            color: var(--text) !important;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


inject_css_light()

# =========================
# 2) Config (실데이터 스키마 기준)
# =========================
RECO_PRODUCT_MAP = {
    "요구불예금좌수": 0,
    "거치식예금좌수": 1,
    "적립식예금좌수": 2,
    "수익증권좌수": 3,
    "신탁좌수": 4,
    "퇴직연금좌수": 5,
    "여신_운전자금대출좌수": 6,
    "여신_시설자금대출좌수": 7,
    "신용카드개수": 8,
    "외환_수출실적거래건수": 9,
    "외환_수입실적거래건수": 10,
}

DEFAULT_RADAR_AMOUNT_COLS = [
    "창구거래금액", "인터넷뱅킹거래금액", "스마트뱅킹거래금액",
    "폰뱅킹거래금액", "ATM거래금액", "자동이체금액",
    "신용카드사용금액", "체크카드사용금액",
    "외환_수출실적금액", "외환_수입실적금액",
    "요구불예금잔액", "거치식예금잔액", "적립식예금잔액",
    "수익증권잔액", "신탁잔액", "퇴직연금잔액",
    "여신_운전자금대출잔액", "여신_시설자금대출잔액",
]

META_COLS = ["업종_중분류", "사업장_시도", "사업장_시군구", "법인_고객등급", "전담고객여부", "RFMP_Segment"]

# =========================
# 3) Session State (router)
# =========================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "selected_customer_id" not in st.session_state:
    st.session_state.selected_customer_id = None
if "selected_month" not in st.session_state:
    st.session_state.selected_month = None
if "_df_real" not in st.session_state:
    st.session_state["_df_real"] = None

def goto(page: str, customer_id: str | None = None):
    st.session_state.page = page
    if customer_id is not None:
        st.session_state.selected_customer_id = customer_id

# =========================
# 4) Data load (업로드 기반 + cp949 우선)
# =========================
def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "segment" not in df.columns:
        raise ValueError("필수 컬럼 segment 가 없습니다.")
    if "기준년월" not in df.columns:
        raise ValueError("필수 컬럼 기준년월 이 없습니다.")

    df["segment"] = df["segment"].astype(float).astype(int)
    df["customer_id"] = df["segment"].apply(lambda x: f"S{x}")

    df["기준년월_dt"] = pd.to_datetime(df["기준년월"], errors="coerce")

    # 안전 처리
    for c in ["추천상품_top1", "추천상품_top2"]:
        if c not in df.columns:
            df[c] = ""
    if "churn_prob_6m" not in df.columns:
        df["churn_prob_6m"] = np.nan
    for c in ["Score_R", "Score_F", "Score_M", "Score_P"]:
        if c not in df.columns:
            df[c] = np.nan

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    """
    cp949 -> euc-kr -> utf-8-sig -> utf-8 순으로 시도.
    성공한 인코딩을 함께 반환.
    """
    last_err = None
    for enc in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=enc)
            df = _postprocess(df)
            return df, enc
        except Exception as e:
            last_err = e
    raise last_err

# =========================
# 5) UI Utils
# =========================
def kpi_cards(risk_count: int, total: int, avg_risk: float, top1_share: float):
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">현재 이탈 위험 고객</div>
                <div class="kpi-value">{risk_count}명</div>
                <div class="kpi-sub">기준 이상 고객 수</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">전체 고객</div>
                <div class="kpi-value">{total}명</div>
                <div class="kpi-sub">선택 월 기준</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">평균 이탈 확률(6M)</div>
                <div class="kpi-value">{avg_risk:.2f}</div>
                <div class="kpi-sub">churn_prob_6m 평균</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Top1 추천 보유율</div>
                <div class="kpi-value">{top1_share:.0f}%</div>
                <div class="kpi-sub">추천상품_top1 존재 비중</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def risk_bar(value: float, label: str = "6M 이탈확률"):
    if pd.isna(value):
        value = 0.0
    value = float(np.clip(value, 0, 1))
    pct = int(value * 100)
    color = "#ff4d4f" if value >= 0.7 else ("#2563eb" if value <= 0.4 else "#f59e0b")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="min-width:96px; color: rgba(15,23,42,0.65); font-size:12px;">{label}: {value:.2f}</div>
            <div style="flex:1; height:10px; background: rgba(15,23,42,0.10); border-radius:999px; overflow:hidden;">
                <div style="width:{pct}%; height:10px; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def gauge_percent(value_0_1: float, title: str, subtitle: str):
    v = 0.0 if pd.isna(value_0_1) else float(np.clip(value_0_1, 0, 1))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v * 100,
        number={"suffix": "%"},
        title={"text": f"{title}<br><span style='font-size:12px;color:rgba(15,23,42,0.62)'>{subtitle}</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563eb"},
            "bgcolor": "#ffffff",
            "borderwidth": 1,
            "bordercolor": "rgba(15,23,42,0.10)",
        }
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=230,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

def gauge_score(value: float, title: str, subtitle: str, min_v: float = 1, max_v: float = 5):
    v = min_v if pd.isna(value) else float(value)
    v = float(np.clip(v, min_v, max_v))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": ""},
        title={"text": f"{title}<br><span style='font-size:12px;color:rgba(15,23,42,0.62)'>{subtitle}</span>"},
        gauge={
            "axis": {"range": [min_v, max_v]},
            "bar": {"color": "#2563eb"},
            "bgcolor": "#ffffff",
            "borderwidth": 1,
            "bordercolor": "rgba(15,23,42,0.10)",
        }
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=230,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

def radar_amounts(row: pd.Series, df_scope: pd.DataFrame, amount_cols: list[str]):
    cols = [c for c in amount_cols if c in df_scope.columns]
    cols = cols[:10] if len(cols) > 10 else cols
    if len(cols) < 3:
        st.info("레이더 차트를 그리려면 최소 3개 이상의 금액/잔액 컬럼이 필요합니다.")
        return

    denom = {}
    for c in cols:
        s = df_scope[c].astype(float)
        q = np.nanquantile(s, 0.95)
        denom[c] = q if (q is not None and np.isfinite(q) and q > 0) else (np.nanmax(s) if np.nanmax(s) > 0 else 1.0)

    values = []
    for c in cols:
        v = float(row[c]) if (c in row.index and not pd.isna(row[c])) else 0.0
        v = np.log1p(max(v, 0.0))
        d = np.log1p(denom[c])
        values.append(float(np.clip(v / d, 0, 1)))

    values += values[:1]
    labels = cols + [cols[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", name="금액/잔액 프로필"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

def make_comments_real(row: pd.Series) -> list[str]:
    comments = []
    p = row.get("churn_prob_6m", np.nan)
    if pd.isna(p):
        comments.append("이탈확률 결측: 산출 파이프라인/조인 여부 점검 필요")
    else:
        if p >= 0.75:
            comments.append("이탈 위험 매우 높음: 즉시 컨택/원인 진단 우선")
        elif p >= 0.55:
            comments.append("이탈 위험 상승 구간: 리텐션 액션/접점 강화 권장")
        else:
            comments.append("관계 안정 구간: 유지 및 교차판매 검토")

    sP = row.get("Score_P", np.nan)
    sF = row.get("Score_F", np.nan)
    if not pd.isna(sP) and sP <= 2:
        comments.append("P(상품다양성) 낮음: 관계 확장 여지(상품군 제안) 큼")
    elif not pd.isna(sF) and sF <= 2:
        comments.append("F(거래빈도) 낮음: 사용 습관화/채널 활성화 필요")
    else:
        comments.append("관계 지표 균형: 고가치화(상위 상품/한도/투자) 검토 가능")

    t1 = str(row.get("추천상품_top1", "")).strip()
    t2 = str(row.get("추천상품_top2", "")).strip()
    if t1 or t2:
        comments.append(f"추천상품: {t1} / {t2}")
    else:
        comments.append("추천상품 정보 없음: 추천 결과 생성/머지 확인 필요")

    return comments[:3]

# =========================
# 6) Sidebar: 업로드 + 메뉴
# =========================
with st.sidebar:
    st.markdown("## 데이터 업로드")
    uploaded = st.file_uploader("CSV 업로드 (cp949 가능)", type=["csv"])
    st.caption("업로드 후 자동으로 읽습니다. (cp949 → euc-kr → utf-8-sig 순 자동 시도)")

    if uploaded is not None:
        try:
            df_loaded, used_enc = read_uploaded_csv(uploaded.getvalue())
            st.session_state["_df_real"] = df_loaded
            st.session_state["page"] = "dashboard"
            st.session_state["selected_customer_id"] = None
            st.success(f"로드 성공 (encoding: {used_enc}, rows: {len(df_loaded):,}, cols: {df_loaded.shape[1]})")
        except Exception as e:
            st.session_state["_df_real"] = None
            st.error(f"데이터 로드 실패: {e}")

    st.markdown("---")
    st.markdown("## 메뉴")
    if st.button("이탈 위험 모니터링(메인)", use_container_width=True):
        goto("dashboard")
    if st.button("추천상품별 고객 리스트", use_container_width=True):
        goto("list")
    if st.button("고객 상세", use_container_width=True, disabled=st.session_state.selected_customer_id is None):
        goto("detail")

    st.markdown("---")
    st.caption("배포형 UI: 업로드 기반으로 즉시 사용 가능합니다.")

# 업로드 전이면 stop
if st.session_state["_df_real"] is None:
    st.warning("좌측 사이드바에서 CSV 파일을 업로드해주세요.")
    st.stop()

df_all = st.session_state["_df_real"].copy()

# =========================
# 7) 월 선택 (업로드된 데이터 기준)
# =========================
months = df_all["기준년월_dt"].dropna().sort_values().unique()
if len(months) == 0:
    st.error("기준년월 파싱 실패: 기준년월 형식을 확인하세요.")
    st.stop()

if st.session_state.selected_month is None or st.session_state.selected_month not in months:
    st.session_state.selected_month = months[-1]

with st.sidebar:
    st.markdown("## 기준월")
    selected_month = st.selectbox(
        "기준년월 선택",
        options=list(months),
        index=list(months).index(st.session_state.selected_month),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m"),
    )
    st.session_state.selected_month = selected_month

df = df_all[df_all["기준년월_dt"] == st.session_state.selected_month].copy()
if df.empty:
    st.warning("선택한 기준년월에 해당하는 데이터가 없습니다. 다른 월을 선택하세요.")
    st.stop()

# =========================
# 8) Pages
# =========================
def page_dashboard(df: pd.DataFrame):
    st.markdown('<div class="big-title">이탈 위험 모니터링</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">실데이터 기반: 6개월 이탈확률(churn_prob_6m) 중심</div>', unsafe_allow_html=True)

    st.markdown("### 이탈 위험 기준 (churn_prob_6m threshold)")
    threshold = st.slider("", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

    flagged = df[df["churn_prob_6m"].fillna(0) >= threshold].copy()
    top1_share = 100.0 * (df["추천상품_top1"].astype(str).str.strip() != "").mean()

    kpi_cards(
        risk_count=int(len(flagged)),
        total=int(len(df)),
        avg_risk=float(df["churn_prob_6m"].fillna(0).mean()),
        top1_share=float(top1_share),
    )

    st.markdown("")
    col1, col2 = st.columns([1.2, 1.0], gap="large")

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 위험 고객 Top 10 (이탈확률 기준 정렬)")

        top = flagged.sort_values(["churn_prob_6m", "Raw_M"], ascending=[False, False]).head(10)
        if top.empty:
            st.info("현재 기준에서 위험 고객이 없습니다. 기준을 낮춰보세요.")
        else:
            for _, r in top.iterrows():
                seg = int(r["segment"])
                cust_id = f"S{seg}"

                st.markdown(
                    f"""
                    <div class="cust-card">
                        <div class="cust-grid">
                            <div>
                                <b>SEG {seg}</b><span class="tag">{cust_id}</span><br/>
                                <span class="muted">{r.get('업종_중분류','-')} · {r.get('사업장_시도','-')} {r.get('사업장_시군구','-')}</span>
                            </div>
                            <div>
                    """,
                    unsafe_allow_html=True
                )
                risk_bar(float(r.get("churn_prob_6m", 0.0)))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div><span class='muted'>추천: <b>{r.get('추천상품_top1','')}</b> / <b>{r.get('추천상품_top2','')}</b></span></div>",
                    unsafe_allow_html=True
                )

                btn = st.button("상세 보기", key=f"dash_detail_{cust_id}")
                st.markdown("</div></div>", unsafe_allow_html=True)

                if btn:
                    st.session_state.selected_customer_id = cust_id
                    goto("detail", cust_id)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 분포 요약")
        st.caption("이탈확률 분포와 기준선 위치를 함께 확인합니다.")

        hist = go.Figure()
        hist.add_trace(go.Histogram(x=df["churn_prob_6m"].fillna(0), nbinsx=20))
        hist.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="#ff4d4f")
        hist.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=360,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis=dict(title="churn_prob_6m"),
            yaxis=dict(title="고객 수"),
        )
        st.plotly_chart(hist, use_container_width=True)

        st.markdown("#### 운영 인사이트(요약)")
        st.write(
            f"- 기준 {threshold:.2f}에서 **{len(flagged)}개 세그먼트**가 관리 대상입니다.\n"
            f"- 평균 이탈확률은 **{df['churn_prob_6m'].fillna(0).mean():.2f}** 입니다.\n"
            f"- 추천상품별 리스트에서 상품별 대상자 목록을 확인하세요."
        )
        st.markdown("</div>", unsafe_allow_html=True)

def page_list(df: pd.DataFrame):
    st.markdown('<div class="big-title">추천상품별 고객 리스트</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">추천상품(top1/top2)에 특정 상품이 포함된 고객을 필터링합니다.</div>', unsafe_allow_html=True)

    colA, colB = st.columns([1.2, 1.0], gap="large")
    with colA:
        product = st.selectbox("추천상품 선택", list(RECO_PRODUCT_MAP.keys()), index=0)
        st.caption(f"상품 인덱스: {RECO_PRODUCT_MAP[product]}")
    with colB:
        search = st.text_input("검색(SEG 번호)", value="")

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2], gap="large")
    with c1:
        min_risk = st.slider("이탈확률 최소", 0.0, 1.0, 0.40, 0.01)
    with c2:
        rfmp_type = st.selectbox("RFMP 세그먼트(옵션)", ["전체"] + sorted(df["RFMP_Segment"].dropna().astype(str).unique().tolist()))
    with c3:
        topk = st.selectbox("표시 개수", [30, 50, 100], index=0)

    t1 = df["추천상품_top1"].astype(str)
    t2 = df["추천상품_top2"].astype(str)
    view = df[(t1 == product) | (t2 == product)].copy()

    view = view[view["churn_prob_6m"].fillna(0) >= min_risk]
    if rfmp_type != "전체":
        view = view[view["RFMP_Segment"].astype(str) == rfmp_type]

    if search.strip():
        s = search.strip().replace("S", "").strip()
        view = view[view["segment"].astype(int).astype(str).str.contains(s, case=False, na=False)]

    view = view.sort_values(["churn_prob_6m", "Raw_M"], ascending=[False, False])

    st.markdown(f"### 타겟 고객 ({len(view)}명)")
    st.caption("상세 보기 버튼으로 고객 상세 페이지로 이동합니다.")

    if view.empty:
        st.info("조건에 맞는 고객이 없습니다. 필터를 완화해보세요.")
        return

    for _, r in view.head(int(topk)).iterrows():
        seg = int(r["segment"])
        cust_id = f"S{seg}"

        st.markdown(
            f"""
            <div class="cust-card">
                <div class="cust-grid">
                    <div>
                        <b>SEG {seg}</b><span class="tag">{cust_id}</span><br/>
                        <span class="muted">{r.get('업종_중분류','-')} · {r.get('사업장_시도','-')} {r.get('사업장_시군구','-')}</span>
                    </div>
                    <div>
            """,
            unsafe_allow_html=True
        )
        risk_bar(float(r.get("churn_prob_6m", 0.0)))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div><span class='muted'>추천: <b>{r.get('추천상품_top1','')}</b> / <b>{r.get('추천상품_top2','')}</b></span></div>",
            unsafe_allow_html=True
        )

        btn = st.button("상세 보기", key=f"list_detail_{cust_id}")
        st.markdown("</div></div>", unsafe_allow_html=True)
        if btn:
            st.session_state.selected_customer_id = cust_id
            goto("detail", cust_id)
            st.rerun()

def page_detail(df: pd.DataFrame):
    cid = st.session_state.selected_customer_id
    if cid is None:
        st.warning("선택된 고객이 없습니다. 리스트에서 고객을 선택해주세요.")
        return

    seg = int(str(cid).replace("S", "").strip())
    row_df = df[df["segment"].astype(int) == seg]
    if row_df.empty:
        st.warning("선택한 고객이 현재 선택 월 데이터에 없습니다. 다른 월을 선택하거나 리스트에서 재선택하세요.")
        return

    row = row_df.iloc[0]

    st.markdown('<div class="big-title">고객 상세 (포켓몬 정보창)</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        meta_lines = []
        for c in META_COLS:
            if c in df.columns:
                meta_lines.append(f"{c}: <b>{row.get(c, '-')}</b>")
        meta_html = " · ".join(meta_lines) if meta_lines else "메타 정보"

        st.markdown(
            f"""
            <div class="panel">
                <div style="display:flex; align-items:flex-end; gap:10px;">
                    <div style="font-size:44px; font-weight:900;">SEG {seg}</div>
                    <div class="tag" style="font-size:14px; padding:6px 12px;">{cid}</div>
                </div>
                <div class="muted" style="margin-top:8px;">{meta_html}</div>
                <div class="muted" style="margin-top:8px;">
                    추천상품: <b>{row.get('추천상품_top1','')}</b> / <b>{row.get('추천상품_top2','')}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("6M 이탈확률", f"{float(row.get('churn_prob_6m', 0.0)):.2f}")
        c2.metric("Raw_M", f"{float(row.get('Raw_M', 0.0)):.1f}")
        c3.metric("RFMP", f"{row.get('RFMP_Segment','-')}")
        st.caption("※ 상세 지표는 사내 산출 기준으로 표시합니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([1.05, 1.45], gap="large")

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 금액/잔액 레이더 (요약)")

        available_amount_cols = [c for c in DEFAULT_RADAR_AMOUNT_COLS if c in df.columns]
        selected_cols = st.multiselect(
            "레이더 축(금액/잔액 컬럼) 선택",
            options=available_amount_cols,
            default=available_amount_cols[:8] if len(available_amount_cols) >= 8 else available_amount_cols
        )

        radar_amounts(row, df, selected_cols if selected_cols else available_amount_cols)
        st.caption("스케일: log1p 후 선택월 95% 분위 기준 0~1 정규화")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 이탈확률 + RFMP 점수")

        g1, g2 = st.columns(2)
        with g1:
            gauge_percent(float(row.get("churn_prob_6m", 0.0)), "6개월 이탈확률", "모델 산출값")
        with g2:
            gauge_score(float(row.get("Score_R", np.nan)), "Score_R", "최근성 점수(1~5)")

        g3, g4 = st.columns(2)
        with g3:
            gauge_score(float(row.get("Score_F", np.nan)), "Score_F", "빈도 점수(1~5)")
        with g4:
            gauge_score(float(row.get("Score_P", np.nan)), "Score_P", "다양성 점수(1~5)")

        st.markdown("---")
        st.markdown("### 코멘트(자동 요약)")
        for msg in make_comments_real(row):
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(15,23,42,0.10);
                            background: rgba(37,99,235,0.08);
                            border-radius: 14px;
                            padding: 12px 14px;
                            margin-bottom: 10px;">
                    <b style="color:#0b1220;">{msg}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("← 리스트로 돌아가기", use_container_width=True):
            goto("list")
            st.rerun()
    with b2:
        if st.button("대시보드로", use_container_width=True):
            goto("dashboard")
            st.rerun()

# =========================
# 9) Router
# =========================
if st.session_state.page == "dashboard":
    page_dashboard(df)
elif st.session_state.page == "list":
    page_list(df)
elif st.session_state.page == "detail":
    page_detail(df)
else:
    page_dashboard(df)

