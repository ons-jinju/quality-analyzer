import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

GITHUB_REPO = "ons-jinju/quality-analyzer"
GITHUB_PATH = "_quality_data.xlsx"


def fetch_from_github() -> bytes | None:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    try:
        resp = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"}, timeout=10)
        if resp.status_code == 200:
            return base64.b64decode(resp.json()["content"])
    except Exception:
        pass
    return None


@st.cache_resource
def shared_store():
    """모든 기기·세션이 공유하는 서버 메모리 저장소"""
    return {"file_bytes": None}


def commit_to_github(file_bytes: bytes) -> bool:
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    existing = requests.get(url, headers=headers, timeout=10)
    sha = existing.json().get("sha") if existing.status_code == 200 else None
    payload = {
        "message": "Update quality data",
        "content": base64.b64encode(file_bytes).decode(),
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(url, json=payload, headers=headers, timeout=30)
    return resp.status_code in (200, 201)

st.set_page_config(page_title="CellScope", page_icon="📡", layout="wide")

st.markdown("""
<style>
@media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.5rem 2rem 0.5rem !important; }
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        flex: 1 1 100% !important; min-width: 100% !important;
    }
    button[data-baseweb="tab"] { font-size: 0.85rem !important; }
    div[data-testid="metric-container"] label { font-size: 0.75rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    div[data-testid="stDataFrame"] { font-size: 0.78rem !important; }
}
</style>
""", unsafe_allow_html=True)

KPI_CONFIG = {
    "rrc_succ_rate":           {"name": "RRC 연결 성공률",   "unit": "%",    "higher_better": True},
    "erab_succ_rate":          {"name": "E-RAB 연결 성공률", "unit": "%",    "higher_better": True},
    "cd_rate":                 {"name": "CD율",              "unit": "%",    "higher_better": False},
    "pmho_succ_rate":          {"name": "HO 성공률",         "unit": "%",    "higher_better": True},
    "rd_acc_succ_rate":        {"name": "랜덤접속 성공률",   "unit": "%",    "higher_better": True},
    "rre_occur_rate":          {"name": "RRE 발생률",        "unit": "%",    "higher_better": False},
    "pmpdcpvoldldrb_thruput":  {"name": "DL 처리량",        "unit": "kbps", "higher_better": True},
    "pmpdcppkt_dl_err_rate":   {"name": "DL 패킷 에러율",   "unit": "%",    "higher_better": False},
    "pmpdcppkt_ul_err_rate":   {"name": "UL 패킷 에러율",   "unit": "%",    "higher_better": False},
    "up_loss_rate":            {"name": "UL 손실률",         "unit": "%",    "higher_better": False},
    "dl_loss_rate":            {"name": "DL 손실률",         "unit": "%",    "higher_better": False},
    "pmsinrpucchdistr":        {"name": "SINR",             "unit": "",     "higher_better": True},
    "mimo_rate":               {"name": "MIMO율",           "unit": "%",    "higher_better": True},
    "tot_cei_value":           {"name": "총 CEI",           "unit": "",     "higher_better": True},
    "lte_cei_value":           {"name": "LTE CEI",          "unit": "",     "higher_better": True},
    "pmprbuseddlavg":                  {"name": "DL PRB 사용률",  "unit": "%",  "higher_better": False},
    "pmradiorecinterferencepwr":       {"name": "PUSCH 간섭",     "unit": "dBm", "higher_better": False},
    "pmradiorecinterferencepwrpucch":  {"name": "PUCCH 간섭",     "unit": "dBm", "higher_better": False},
}

INVALID = -9999999


def load_data(file) -> pd.DataFrame:
    buf = file if isinstance(file, io.BytesIO) else io.BytesIO(file.getvalue())
    sheets = pd.read_excel(buf, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.replace(INVALID, np.nan, inplace=True)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["dt"])
    if "cell_num" in df.columns:
        df["cell_num"] = pd.to_numeric(df["cell_num"], errors="coerce").astype("Int64")
    return df


def analyze(sub: pd.DataFrame, base_date: pd.Timestamp, compare_date: pd.Timestamp) -> pd.DataFrame:
    kpi_cols = [c for c in KPI_CONFIG if c in sub.columns]
    daily = sub.groupby("dt")[kpi_cols].mean().sort_index()
    if len(daily) < 2:
        return pd.DataFrame()

    rows = []
    for col in kpi_cols:
        series = daily[col].dropna()
        if len(series) < 2:
            continue
        if base_date not in series.index or compare_date not in series.index:
            continue
        cfg = KPI_CONFIG[col]
        hb = cfg["higher_better"]
        v_base    = series[base_date]
        v_compare = series[compare_date]
        delta     = v_compare - v_base
        delta_pct = (delta / abs(v_base) * 100) if v_base != 0 else np.nan
        slope     = np.polyfit(np.arange(len(series)), series.values, 1)[0]
        f_worse   = delta < 0 if hb else delta > 0
        t_worse   = slope < 0 if hb else slope > 0
        rows.append({
            "KPI": cfg["name"], "col": col, "단위": cfg["unit"], "higher_better": hb,
            f"기준({base_date.strftime('%m/%d')})":    round(v_base, 2),
            f"비교({compare_date.strftime('%m/%d')})": round(v_compare, 2),
            "변화량": round(delta, 2),
            "변화율(%)": round(delta_pct, 1) if not np.isnan(delta_pct) else "-",
            "추세기울기": slope,
            "기준대비 악화": f_worse, "추세 악화": t_worse,
            "둘다 악화": f_worse and t_worse,
        })
    return pd.DataFrame(rows)


def trend_icon(slope, hb):
    if hb:
        return "📉 지속하락" if slope < -0.01 else ("📈 상승중" if slope > 0.01 else "➡️ 보합")
    return "📈 지속상승" if slope > 0.01 else ("📉 하락중" if slope < -0.01 else "➡️ 보합")


def is_bad(row):
    return row["기준대비 악화"] or row["추세 악화"]


def make_chart(daily, col, kpi_name, unit, color="#3498db"):
    series = daily[col].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=8),
    ))
    if len(series) >= 2:
        x_num = np.arange(len(series))
        s, i = np.polyfit(x_num, series.values, 1)
        fig.add_trace(go.Scatter(
            x=series.index, y=s * x_num + i, mode="lines",
            line=dict(color="#e67e22", dash="dash", width=1.5),
        ))
    fig.update_layout(
        title=dict(text=f"{kpi_name} ({unit})" if unit else kpi_name, font=dict(size=12)),
        height=220, margin=dict(l=10, r=10, t=38, b=25), showlegend=False,
        xaxis=dict(tickformat="%m/%d", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def render_html_table(rows: list[dict]) -> None:
    """st.dataframe 대신 정적 HTML 테이블 — 모바일 터치 스크롤 간섭 없음."""
    if not rows:
        return
    headers = list(rows[0].keys())
    th = "".join(
        f"<th style='padding:7px 6px;background:#f5f5f5;border-bottom:2px solid #ddd;"
        f"font-size:0.78rem;white-space:nowrap;text-align:left;'>{h}</th>"
        for h in headers
    )
    body = ""
    for idx, r in enumerate(rows):
        bg = "background:#fafafa;" if idx % 2 == 0 else ""
        tds = ""
        for h in headers:
            v = r[h]
            color = ""
            if h == "상태":
                if "심각" in str(v):   color = "color:#e74c3c;font-weight:700;"
                elif "악화" in str(v): color = "color:#e67e22;font-weight:700;"
                elif "하락" in str(v): color = "color:#f39c12;font-weight:700;"
                else:                   color = "color:#27ae60;font-weight:700;"
            tds += (
                f"<td style='padding:7px 6px;border-bottom:1px solid #eee;"
                f"font-size:0.82rem;white-space:nowrap;{bg}{color}'>{v}</td>"
            )
        body += f"<tr>{tds}</tr>"

    html = (
        "<div style='overflow-x:auto;-webkit-overflow-scrolling:touch;max-height:420px;overflow-y:auto;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ── 메인 ─────────────────────────────────────────────────────────────────────
st.markdown("## 📡 CellScope")

# ── 데이터 소스: 공유 메모리 → GitHub → 업로드 ───────────────────────────────
store = shared_store()

# 공유 메모리에 없으면 GitHub에서 가져오기
if store["file_bytes"] is None:
    with st.spinner("데이터 불러오는 중..."):
        store["file_bytes"] = fetch_from_github()

if store["file_bytes"]:
    col_msg, col_btn = st.columns([6, 1])
    col_msg.success("✅ 데이터 로드됨 — 모바일에서도 자동 적용됩니다.")
    if col_btn.button("🔄", help="새로고침"):
        store["file_bytes"] = None
        st.rerun()

    with st.expander("📂 데이터 업데이트 (새 파일로 교체)"):
        uploaded = st.file_uploader("새 엑셀 파일 업로드 (.xlsx)", type=["xlsx"], key="updater")
        if uploaded:
            raw = uploaded.getvalue()
            with st.spinner("저장 중..."):
                ok = commit_to_github(raw)
            store["file_bytes"] = raw   # 공유 메모리 즉시 반영 → 모바일도 바로 적용
            if ok:
                st.success("✅ 업로드 완료! 모든 기기에 즉시 반영됩니다.")
            else:
                st.warning("⚠️ GitHub 저장 실패 (이번 세션만 유지됩니다). GITHUB_TOKEN을 확인하세요.")
else:
    st.warning("📂 저장된 데이터가 없습니다. 엑셀 파일을 업로드해 주세요.")
    uploaded = st.file_uploader("품질 데이터 엑셀 파일 업로드 (.xlsx)", type=["xlsx"], key="first_upload")
    if not uploaded:
        st.stop()
    raw = uploaded.getvalue()
    with st.spinner("저장 중..."):
        ok = commit_to_github(raw)
    store["file_bytes"] = raw
    if ok:
        st.success("✅ 업로드 완료! 모든 기기에서 자동으로 로드됩니다.")
    else:
        st.warning("⚠️ GitHub 저장 실패. 이번 세션 동안만 유지됩니다.")

with st.spinner("데이터 로딩 중..."):
    df = load_data(io.BytesIO(store["file_bytes"]))

if "eqp_nm" not in df.columns:
    st.error("eqp_nm 컬럼을 찾을 수 없습니다.")
    st.stop()

dates_avail = sorted(df["dt"].dropna().unique())
all_stations = sorted(df["eqp_nm"].dropna().unique().tolist())

# ── 사이드바 ─────────────────────────────────────────────────────────────────
sb = st.sidebar
sb.header("🔧 분석 설정")

# session_state 초기화
for key, default in [
    ("confirmed_stations", []),
    ("confirmed_cells", []),
    ("all_cells_checked", True),
    ("prev_stations", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── ① 국소 선택 ──────────────────────────────────────────────────────────────
sb.markdown("**① 국소 선택**")
pending_stations = sb.multiselect(
    "국소 검색 또는 선택",
    options=all_stations,
    default=st.session_state.confirmed_stations or ([all_stations[0]] if all_stations else []),
    placeholder="국소명 검색...",
    label_visibility="collapsed",
)
if sb.button("✅ 국소 확인", use_container_width=True, type="primary"):
    if pending_stations != st.session_state.prev_stations:
        st.session_state.confirmed_cells = []        # 국소 바뀌면 cell 초기화
        st.session_state.prev_stations = pending_stations
    st.session_state.confirmed_stations = pending_stations

selected_stations = st.session_state.confirmed_stations

if not selected_stations:
    sb.info("국소 선택 후 확인을 눌러주세요.")
    st.info("👈 왼쪽에서 국소를 선택하고 **확인** 버튼을 눌러주세요.")
    st.stop()

sb.caption(f"✔ 선택됨: {', '.join(selected_stations)}")

# ── ② Cell 선택 ──────────────────────────────────────────────────────────────
df_filtered_station = df[df["eqp_nm"].isin(selected_stations)]

if "cell_num" in df.columns and "ru_name" in df.columns:
    cell_ru = (
        df_filtered_station[["cell_num", "ru_name"]]
        .drop_duplicates().dropna(subset=["cell_num"]).sort_values("cell_num")
    )
    cell_options = cell_ru["cell_num"].dropna().unique().tolist()
    cell_labels = {
        int(r["cell_num"]): f"Cell {int(r['cell_num'])}  |  {r['ru_name']}"
        for _, r in cell_ru.iterrows() if pd.notna(r["cell_num"])
    }
else:
    cell_options, cell_labels = [], {}

sb.markdown("---")
sb.markdown("**② Cell 선택**")

all_cells_check = sb.checkbox("전체 Cell", value=st.session_state.all_cells_checked)
st.session_state.all_cells_checked = all_cells_check

if not all_cells_check:
    pending_cells = sb.multiselect(
        "Cell 선택",
        options=cell_options,
        format_func=lambda c: cell_labels.get(int(c), f"Cell {c}"),
        default=st.session_state.confirmed_cells or cell_options[:1],
        placeholder="Cell 선택...",
        label_visibility="collapsed",
    )
    if sb.button("✅ Cell 확인", use_container_width=True, type="primary"):
        st.session_state.confirmed_cells = pending_cells
    selected_cells = st.session_state.confirmed_cells or cell_options
else:
    selected_cells = cell_options
    st.session_state.confirmed_cells = cell_options

# 선택 Cell / RU 요약
if cell_options:
    sb.markdown("---")
    sb.markdown("**📋 Cell / RU**")
    for c in sorted(selected_cells):
        sb.caption(cell_labels.get(int(c), f"Cell {c}"))

# ── ③ 날짜 선택 ──────────────────────────────────────────────────────────────
sb.markdown("---")
sb.markdown("**③ 날짜 비교 설정**")

date_labels = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates_avail]
date_map    = {label: pd.Timestamp(d) for label, d in zip(date_labels, dates_avail)}

base_label    = sb.selectbox("기준일", options=date_labels, index=0)
compare_label = sb.selectbox("비교일", options=date_labels, index=len(date_labels) - 1)

base_date    = date_map[base_label]
compare_date = date_map[compare_label]

if base_date == compare_date:
    sb.warning("기준일과 비교일이 같습니다.")

# 기타 설정
sb.markdown("---")
show_only_bad = sb.checkbox("악화 항목만 표시", value=True)
sb.caption(f"데이터 기간: {date_labels[0]} ~ {date_labels[-1]} (총 {len(date_labels)}일)")

# ── 데이터 필터링 ─────────────────────────────────────────────────────────────
if selected_cells and "cell_num" in df.columns:
    sub = df_filtered_station[df_filtered_station["cell_num"].isin(selected_cells)].copy()
else:
    sub = df_filtered_station.copy()

if sub.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ── 선택 정보 요약 ────────────────────────────────────────────────────────────
with st.expander("📋 선택 국소 / Cell / RU 정보", expanded=False):
    info_cols = [c for c in ["eqp_nm", "cell_num", "ru_name"] if c in sub.columns]
    if info_cols:
        info_df = (
            sub[info_cols].drop_duplicates()
            .sort_values(["eqp_nm", "cell_num"] if "cell_num" in info_cols else ["eqp_nm"])
            .rename(columns={"eqp_nm": "국소명", "cell_num": "Cell No.", "ru_name": "RU Name"})
            .reset_index(drop=True)
        )
        render_html_table(info_df.to_dict("records"))

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📉 품질 변화 분석", "🔍 항목별 상세 보기"])

with tab1:
    if base_date == compare_date:
        st.warning("기준일과 비교일을 다르게 선택해 주세요.")
        st.stop()
    result = analyze(sub, base_date, compare_date)
    if result.empty:
        st.warning("날짜 데이터가 2개 이상 필요합니다.")
        st.stop()

    bad  = result[result.apply(is_bad, axis=1)]
    good = result[~result.apply(is_bad, axis=1)]

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 KPI", len(result))
    c2.metric("악화 항목", len(bad), delta=f"-{len(bad)}개", delta_color="inverse")
    c3.metric("정상 항목", len(good))

    st.markdown("---")
    display_df = bad if show_only_bad else result
    first_col  = [c for c in result.columns if "기준(" in c][0]
    last_col   = [c for c in result.columns if "비교(" in c][0]

    if display_df.empty:
        st.success("악화된 KPI가 없습니다! 모든 지표가 양호합니다.")
    else:
        st.markdown(f"### {'⚠️ 악화 항목' if show_only_bad else '전체 KPI 현황'} ({len(display_df)}개)")
        rows = []
        for _, row in display_df.iterrows():
            if row["둘다 악화"]:       status = "🔴 심각"
            elif row["기준대비 악화"]: status = "🟠 악화"
            elif row["추세 악화"]:    status = "🟡 하락추세"
            else:                      status = "🟢 양호"
            dp = row["변화율(%)"]
            rows.append({
                "상태": status, "KPI": row["KPI"], "단위": row["단위"],
                first_col: row[first_col], last_col: row[last_col],
                "변화율": f"{dp:+.1f}%" if isinstance(dp, float) else dp,
                "추세": trend_icon(row["추세기울기"], row["higher_better"]),
            })
        render_html_table(rows)

    worst = result[result["둘다 악화"]].head(6)
    if not worst.empty:
        st.markdown("---")
        st.markdown("### 📉 심각 악화 항목 추세 차트")
        kpi_chart_cols = worst["col"].tolist()
        daily = sub.groupby("dt")[kpi_chart_cols].mean().sort_index()
        for _, row in worst.iterrows():
            if row["col"] in daily.columns:
                st.plotly_chart(
                    make_chart(daily, row["col"], row["KPI"], row["단위"], "#e74c3c"),
                    use_container_width=True,
                )

with tab2:
    kpi_options = {cfg["name"]: col for col, cfg in KPI_CONFIG.items() if col in df.columns}
    selected_kpi_names = st.multiselect(
        "분석할 KPI 항목 선택",
        options=list(kpi_options.keys()),
        default=list(kpi_options.keys())[:3],
    )
    if not selected_kpi_names:
        st.info("위에서 KPI 항목을 선택해 주세요.")
    else:
        selected_kpi_cols = [kpi_options[n] for n in selected_kpi_names]
        daily_detail = sub.groupby("dt")[selected_kpi_cols].mean().sort_index()
        view_type = st.radio("보기 방식", ["그래프", "표", "그래프 + 표"], horizontal=True)

        if view_type in ("그래프", "그래프 + 표"):
            for name in selected_kpi_names:
                ck = kpi_options[name]
                st.plotly_chart(
                    make_chart(daily_detail, ck, name, KPI_CONFIG[ck]["unit"]),
                    use_container_width=True,
                )
        if view_type in ("표", "그래프 + 표"):
            st.markdown("#### 날짜별 수치")
            tbl = daily_detail.copy()
            tbl.index = tbl.index.strftime("%Y-%m-%d")
            tbl.columns = selected_kpi_names
            tbl = tbl.round(2).reset_index().rename(columns={"index": "날짜"})
            render_html_table(tbl.to_dict("records"))

st.markdown("---")
st.caption(
    f"국소: **{', '.join(selected_stations)}** | "
    f"Cell: **{'전체' if (not cell_options or select_all_cells) else str(sorted(selected_cells))}** | "
    f"기간: {pd.Timestamp(dates_avail[0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(dates_avail[-1]).strftime('%Y-%m-%d')}"
)
