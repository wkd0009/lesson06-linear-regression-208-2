import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외
SLIDER_MIN, SLIDER_MAX = 1900, 2100

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data
def load_yearly_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만인 해 제외
    return yearly.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세를 회귀 직선으로 만들고, 원하는 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("조건에 맞는 해가 2개 미만이라 회귀 직선을 만들 수 없어요.")
    st.stop()

x = yearly["연도"].to_numpy()
y = yearly["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])
start_year, end_year, n_years = int(x.min()), int(x.max()), len(yearly)

# ---- 직선을 만든 데이터 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

st.caption(
    f"기준: {LAST_YEAR}년까지의 데이터, 관측일 {MIN_DAYS}일 이상인 해만 사용 · "
    f"회귀 직선: 기온 = {slope:.4f} × 연도 + ({intercept:.2f}) "
    f"(10년에 약 {slope * 10:+.2f}℃)"
)

# ---- 연도 슬라이더 & 예상 기온 ----
st.divider()
year = st.slider("예상하고 싶은 연도", SLIDER_MIN, SLIDER_MAX, 2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:1.2rem; opacity:0.7;">{year}년 서울의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.1f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if year < start_year or year > end_year:
    st.info(
        f"선택한 {year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이에요. "
        "기간 밖의 값은 추세를 그대로 늘려 계산한 것이라 불확실성이 커요."
    )

# ---- 산점도 + 회귀 직선 ----
line_x = np.array([min(start_year, year), max(end_year, year)])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        hovertemplate="%{x}년<br>%{y:.2f}℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f}℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온과 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"관측일수": "관측일수(일)"}).round(2),
        use_container_width=True,
        hide_index=True,
    )
