
import math

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write("영화의 흥행 특성을 바탕으로 세 가지 유형으로 분류합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

FEATURES = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "longrun_index",
]

LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수",
}

TYPE_NAMES = ["㉮", "㉯", "㉰"]


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw = load_data()
except Exception as e:
    st.error("CSV 파일을 불러오지 못했습니다.")
    st.code(str(e))
    st.info("인터넷 연결과 데이터 URL을 확인한 뒤 다시 실행하세요.")
    st.stop()


# --------------------------------------------------
# 전처리
# --------------------------------------------------
required = [
    "movieNm",
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

missing = [c for c in required if c not in raw.columns]

if missing:
    st.error("데이터에 필요한 열이 없습니다.")
    st.write(missing)
    st.stop()

df = raw.copy()

for col in [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

total_count = len(df)

# 필요한 수치가 없거나 첫 주 관객이 0 이하인 행 제외
df = df.dropna(
    subset=[
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
).copy()

df = df[
    (df["first_scrn"] > 0)
    & (df["first_week_audi"] > 0)
    & (df["total_audi"] > 0)
].copy()

# 파생 변수
df["longrun_index"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

df["log_first_scrn"] = df["first_scrn"].map(math.log10)
df["log_total_audi"] = df["total_audi"].map(math.log10)

df = df.dropna(subset=FEATURES).copy()

# 전체 편수 / 분석 편수
st.markdown(
    f"**전체 영화 {total_count:,}편 · "
    f"분류 가능한 영화 {len(df):,}편**"
)

if len(df) < 3:
    st.error("유효한 영화가 3편 미만이라 군집화를 진행할 수 없습니다.")
    st.stop()


# --------------------------------------------------
# 군집화 속성 선택
# --------------------------------------------------
st.sidebar.header("분석 설정")

selected = st.sidebar.multiselect(
    "군집화에 사용할 속성",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: LABELS[x],
)

if len(selected) < 2:
    st.warning("군집화할 속성을 2개 이상 선택하세요.")
    st.stop()

# 표준화 및 K-평균
scaler = StandardScaler()
X = scaler.fit_transform(df[selected])

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
df["cluster"] = model.fit_predict(X)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

name_map = {
    cluster_id: TYPE_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}

df["유형"] = df["cluster"].map(name_map)


# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------
st.subheader("📊 2차원 산점도")

left, right = st.columns(2)

with left:
    x_axis = st.selectbox(
        "가로축",
        selected,
        format_func=lambda x: LABELS[x],
        key="x2",
    )

with right:
    y_axis = st.selectbox(
        "세로축",
        selected,
        index=min(1, len(selected) - 1),
        format_func=lambda x: LABELS[x],
        key="y2",
    )

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="유형",
    category_orders={"유형": TYPE_NAMES},
    hover_name="movieNm",
    labels={
        x_axis: LABELS[x_axis],
        y_axis: LABELS[y_axis],
        "유형": "영화 유형",
    },
    title="영화 유형 분포",
)

fig2.update_traces(marker={"size": 7, "opacity": 0.75})
st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.subheader("🧊 3차원 산점도")

if len(selected) < 3:
    st.info("3차원 산점도는 군집화 속성을 3개 이상 선택하면 표시됩니다.")
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축",
            selected,
            index=0,
            format_func=lambda x: LABELS[x],
            key="x3",
        )

    with c2:
        y3 = st.selectbox(
            "Y축",
            selected,
            index=1,
            format_func=lambda x: LABELS[x],
            key="y3",
        )

    with c3:
        z3 = st.selectbox(
            "Z축",
            selected,
            index=2,
            format_func=lambda x: LABELS[x],
            key="z3",
        )

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        category_orders={"유형": TYPE_NAMES},
        hover_name="movieNm",
        labels={
            x3: LABELS[x3],
            y3: LABELS[y3],
            z3: LABELS[z3],
            "유형": "영화 유형",
        },
        title="영화 유형 3차원 분포",
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.75}
    )

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 유형별 평균 및 편수
# --------------------------------------------------
st.subheader("📋 유형별 요약")

summary_rows = []

for name in TYPE_NAMES:
    group = df[df["유형"] == name]

    if group.empty:
        continue

    summary_rows.append({
        "영화 유형": name,
        "편수": len(group),
        "스크린 수 평균": group["first_scrn"].mean(),
        "누적 관객 평균": group["total_audi"].mean(),
        "10위권 일수 평균": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["longrun_index"].mean(),
    })

summary = pd.DataFrame(summary_rows)

if not summary.empty:
    display = summary.copy()
    display["스크린 수 평균"] = display["스크린 수 평균"].map(
        lambda x: f"{x:,.1f}"
    )
    display["누적 관객 평균"] = display["누적 관객 평균"].map(
        lambda x: f"{x:,.1f}"
    )
    display["10위권 일수 평균"] = display["10위권 일수 평균"].map(
        lambda x: f"{x:.2f}"
    )
    display["롱런 지수 평균"] = display["롱런 지수 평균"].map(
        lambda x: f"{x:.2f}"
    )

    st.dataframe(display, use_container_width=True, hide_index=True)


# --------------------------------------------------
# 유형별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("🏆 유형별 누적 관객 상위 5편")

for name in TYPE_NAMES:
    group = (
        df[df["유형"] == name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(f"{name} 유형 영화", expanded=True):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
            continue

        top5 = group[
            [
                "movieNm",
                "total_audi",
                "first_week_audi",
                "first_scrn",
                "days_in_top10",
                "longrun_index",
            ]
        ].copy()

        top5.columns = [
            "영화 제목",
            "누적 관객",
            "첫 주 관객",
            "스크린 수",
            "10위권 일수",
            "롱런 지수",
        ]

        for col in ["누적 관객", "첫 주 관객", "스크린 수"]:
            top5[col] = top5[col].map(lambda x: f"{x:,.0f}")

        top5["10위권 일수"] = top5["10위권 일수"].map(
            lambda x: f"{x:.0f}"
        )
        top5["롱런 지수"] = top5["롱런 지수"].map(
            lambda x: f"{x:.2f}"
        )

        st.dataframe(top5, use_container_width=True, hide_index=True)

st.caption(
    "스크린 수와 누적 관객은 상용로그를 군집화에 사용합니다. "
    "요약표의 평균은 원래 단위로 표시합니다."
)
