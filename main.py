
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 특성을 바탕으로 K-평균 군집화를 수행합니다.")


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/"
    "data-share-/main/kobis_movies.csv"
)

REQUIRED_COLUMNS = [
    "movieCd",
    "movieNm",
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

FEATURES = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "longrun_index",
]

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수",
}

ORIGINAL_FEATURES = [
    "first_scrn",
    "total_audi",
    "days_in_top10",
    "longrun_index",
]

ORIGINAL_LABELS = {
    "first_scrn": "스크린 수 평균",
    "total_audi": "누적 관객 평균",
    "days_in_top10": "10위권 일수 평균",
    "longrun_index": "롱런 지수 평균",
}

CLUSTER_NAMES = {0: "㉮", 1: "㉯", 2: "㉰"}


@st.cache_data
def load_data(url):
    data = pd.read_csv(url, encoding="utf-8")

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return data


try:
    raw_df = load_data(DATA_URL)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 데이터 전처리 및 파생 변수 생성
# --------------------------------------------------
df = raw_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
# 첫 주 관객이 0이거나 결측치인 영화는 제외
df["longrun_index"] = (
    df["total_audi"] / df["first_week_audi"].where(
        df["first_week_audi"] != 0
    )
).clip(upper=20)

# 네 속성 중 결측치가 있거나 로그를 계산할 수 없는 영화 제외
# 스크린 수와 누적 관객은 양수여야 상용로그 계산 가능
valid_mask = (
    df[["first_scrn", "total_audi", "days_in_top10",
        "longrun_index"]].notna().all(axis=1)
    & (df["first_week_audi"] > 0)
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
)

df = df.loc[valid_mask].copy()

# 군집화에 사용할 파생 변수
df["log_first_scrn"] = df["first_scrn"].apply(
    lambda x: __import__("math").log10(x)
)
df["log_total_audi"] = df["total_audi"].apply(
    lambda x: __import__("math").log10(x)
)

df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna(subset=FEATURES).copy()

# 전체 영화 수와 분석 대상 영화 수
total_count = len(raw_df)
analyzed_count = len(df)

st.markdown(
    f"**전체 영화: {total_count:,}편 · "
    f"분류한 영화: {analyzed_count:,}편**"
)

if analyzed_count < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()


# --------------------------------------------------
# 4. 분석 속성 선택
# --------------------------------------------------
st.sidebar.header("⚙️ 분석 설정")

selected_features = st.sidebar.multiselect(
    "군집화에 사용할 속성 (2개 이상 권장)",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
)

if len(selected_features) == 0:
    st.warning("군집화에 사용할 속성을 하나 이상 선택하세요.")
    st.stop()

# 선택한 속성 표준화
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[selected_features])

# 난수 고정으로 재실행 시 결과 재현
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
df["cluster"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_audience = df.groupby("cluster")["total_audi"].mean()
sorted_clusters = cluster_audience.sort_values(
    ascending=False
).index.tolist()

cluster_labels = {
    cluster_id: CLUSTER_NAMES[rank]
    for rank, cluster_id in enumerate(sorted_clusters)
}

df["유형"] = df["cluster"].map(cluster_labels)


# --------------------------------------------------
# 5. 2차원 산점도
# --------------------------------------------------
st.subheader("📊 2차원 산점도")

axis_options = selected_features

col_x, col_y = st.columns(2)

with col_x:
    x_axis = st.selectbox(
        "가로축 속성",
        options=axis_options,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="2d_x",
    )

with col_y:
    y_default = min(1, len(axis_options) - 1)
    y_axis = st.selectbox(
        "세로축 속성",
        options=axis_options,
        index=y_default,
        format_func=lambda x: FEATURE_LABELS[x],
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="유형",
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    hover_name="movieNm",
    hover_data={
        "유형": True,
        x_axis: ":.3f",
        y_axis: ":.3f",
    },
    labels={
        x_axis: FEATURE_LABELS[x_axis],
        y_axis: FEATURE_LABELS[y_axis],
        "유형": "영화 유형",
    },
    title="영화 유형별 분포",
)

fig_2d.update_traces(marker={"size": 7, "opacity": 0.75})
fig_2d.update_layout(legend_title_text="영화 유형")

st.plotly_chart(fig_2d, use_container_width=True)


# --------------------------------------------------
# 6. 3차원 산점도
# --------------------------------------------------
st.subheader("🧊 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 표시하려면 군집화에 사용할 속성을 "
        "3개 이상 선택하세요. 왼쪽 설정에서 속성을 추가할 수 있습니다."
    )
else:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="3d_x",
        )

    with col_y3:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="3d_y",
        )

    with col_z3:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        category_orders={"유형": ["㉮", "㉯", "㉰"]},
        hover_name="movieNm",
        hover_data={
            "유형": True,
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f",
        },
        labels={
            x3: FEATURE_LABELS[x3],
            y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3],
            "유형": "영화 유형",
        },
        title="영화 유형별 3차원 분포",
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.75}
    )
    fig_3d.update_layout(
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": FEATURE_LABELS[x3],
            "yaxis_title": FEATURE_LABELS[y3],
            "zaxis_title": FEATURE_LABELS[z3],
        },
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# --------------------------------------------------
# 7. 군집별 편수 및 원래 단위 평균
# --------------------------------------------------
st.subheader("📋 유형별 요약")

summary = (
    df.groupby("유형")
    .agg(
        편수=("movieNm", "size"),
        **{
            ORIGINAL_LABELS["first_scrn"]: ("first_scrn", "mean"),
            ORIGINAL_LABELS["total_audi"]: ("total_audi", "mean"),
            ORIGINAL_LABELS["days_in_top10"]: (
                "days_in_top10", "mean"
            ),
            ORIGINAL_LABELS["longrun_index"]: (
                "longrun_index", "mean"
            ),
        },
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
    .rename(columns={"유형": "영화 유형"})
)

summary = summary.rename(columns={"index": "영화 유형"})

# 평균값을 읽기 쉽게 표시
display_summary = summary.copy()

for col in [
    ORIGINAL_LABELS["first_scrn"],
    ORIGINAL_LABELS["total_audi"],
]:
    display_summary[col] = display_summary[col].map(
        lambda x: f"{x:,.1f}" if pd.notna(x) else "-"
    )

for col in [
    ORIGINAL_LABELS["days_in_top10"],
    ORIGINAL_LABELS["longrun_index"],
]:
    display_summary[col] = display_summary[col].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
    )

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 8. 유형별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("🏆 유형별 누적 관객 상위 영화")

for label in ["㉮", "㉯", "㉰"]:
    group_df = df[df["유형"] == label].sort_values(
        "total_audi", ascending=False
    ).head(5)

    with st.expander(f"{label} 유형 — 누적 관객 상위 5편", expanded=True):
        if group_df.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top5 = group_df[
                ["movieNm", "total_audi", "first_week_audi",
                 "first_scrn", "days_in_top10", "longrun_index"]
            ].copy()

            top5.columns = [
                "영화 제목",
                "누적 관객",
                "첫 주 관객",
                "스크린 수",
                "10위권 일수",
                "롱런 지수",
            ]

            top5["누적 관객"] = top5["누적 관객"].map(
                lambda x: f"{x:,.0f}"
            )
            top5["첫 주 관객"] = top5["첫 주 관객"].map(
                lambda x: f"{x:,.0f}"
            )
            top5["스크린 수"] = top5["스크린 수"].map(
                lambda x: f"{x:,.0f}"
            )
            top5["10위권 일수"] = top5["10위권 일수"].map(
                lambda x: f"{x:,.0f}"
            )
            top5["롱런 지수"] = top5["롱런 지수"].map(
                lambda x: f"{x:.2f}"
            )

            st.dataframe(
                top5,
                use_container_width=True,
                hide_index=True,
            )

st.caption(
    "군집 번호는 누적 관객 평균이 높은 순서대로 ㉮, ㉯, ㉰를 부여했습니다. "
    "로그 변환은 군집화와 산점도에 사용하며, 유형별 평균은 원래 단위로 계산합니다."
)
