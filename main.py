import math
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 페이지 설정 (앱 제목 및 브라우저 탭 아이콘/제목)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


raw_df = load_data()
total_raw_count = len(raw_df)

# 3. 데이터 전처리
required_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df_clean = raw_df.dropna(subset=required_cols).copy()
df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

# 파생 속성 생성
df_clean["log_scrn"] = df_clean["first_scrn"].apply(lambda x: math.log10(x))
df_clean["log_total_audi"] = df_clean["total_audi"].apply(lambda x: math.log10(x))
df_clean["long_run"] = df_clean.apply(
    lambda row: min(row["total_audi"] / row["first_week_audi"], 20.0), axis=1
)

used_count = len(df_clean)

# 전체 편수와 묶은 편수를 화면에 한 줄로 표시
st.write(f"**전체 영화 편수:** {total_raw_count}편 | **분석 대상 영화 편수:** {used_count}편")

st.markdown("---")

# 속성 매핑 Dictionary
feature_dict = {
    "스크린 수 (상용로그)": "log_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run",
}

# 4. 속성 선택 UI (기본값: 네 개 다 선택)
selected_feature_names = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_names) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

selected_cols = [feature_dict[name] for name in selected_feature_names]

# 5. 표준화 & K-Means (K=3) 군집화 (scikit-learn 사용, random_state 고정)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[selected_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean["cluster_raw"] = kmeans.fit_predict(scaled_data)

# 누적 관객 평균(total_audi)이 큰 순서대로 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = (
    df_clean.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False)
)
label_map = {
    cluster_audi_mean.index[0]: "㉮",
    cluster_audi_mean.index[1]: "㉯",
    cluster_audi_mean.index[2]: "㉰",
}
df_clean["cluster"] = df_clean["cluster_raw"].map(label_map)

st.markdown("---")

# 6. 2차원 산점도 (Plotly 사용)
st.subheader("📊 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_2d_name = st.selectbox("가로축 속성:", options=selected_feature_names, index=0)
with col2d_2:
    y_2d_name = st.selectbox(
        "세로축 속성:",
        options=selected_feature_names,
        index=min(1, len(selected_feature_names) - 1),
    )

fig_2d = px.scatter(
    df_clean,
    x=feature_dict[x_2d_name],
    y=feature_dict[y_2d_name],
    color="cluster",
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
    hover_name="movieNm",
    labels={
        feature_dict[x_2d_name]: x_2d_name,
        feature_dict[y_2d_name]: y_2d_name,
        "cluster": "묶음",
    },
)
st.plotly_chart(fig_2d, use_container_width=True)

st.markdown("---")

# 7. 3차원 산점도 (Plotly 사용, 회전 가능 & 점 크게 조절)
st.subheader("🎲 3차원 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 확인하려면 속성을 3개 이상 고르세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_3d_name = st.selectbox("X축 속성:", options=selected_feature_names, index=0, key="3d_x")
    with col3d_2:
        y_3d_name = st.selectbox(
            "Y축 속성:",
            options=selected_feature_names,
            index=min(1, len(selected_feature_names) - 1),
            key="3d_y",
        )
    with col3d_3:
        z_3d_name = st.selectbox(
            "Z축 속성:",
            options=selected_feature_names,
            index=min(2, len(selected_feature_names) - 1),
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df_clean,
        x=feature_dict[x_3d_name],
        y=feature_dict[y_3d_name],
        z=feature_dict[z_3d_name],
        color="cluster",
        category_orders={"cluster": ["㉮", "㉯", "㉰"]},
        hover_name="movieNm",
        labels={
            feature_dict[x_3d_name]: x_3d_name,
            feature_dict[y_3d_name]: y_3d_name,
            feature_dict[z_3d_name]: z_3d_name,
            "cluster": "묶음",
        },
    )
    # 점 크기를 작게 설정
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 8. 묶음별 편수 및 네 속성의 원래 단위 평균 표
st.subheader("📋 묶음별 요약 통계 (원래 단위 평균)")

summary_df = (
    df_clean.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        top10일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

# 표 포맷팅
formatted_summary = summary_df.copy()
formatted_summary["편수"] = formatted_summary["편수"].apply(lambda x: f"{x:,}편")
formatted_summary["스크린수_평균"] = formatted_summary["스크린수_평균"].apply(lambda x: f"{x:,.1f}")
formatted_summary["누적관객_평균"] = formatted_summary["누적관객_평균"].apply(lambda x: f"{x:,.0f}")
formatted_summary["top10일수_평균"] = formatted_summary["top10일수_평균"].apply(lambda x: f"{x:,.1f}")
formatted_summary["롱런지수_평균"] = formatted_summary["롱런지수_평균"].apply(lambda x: f"{x:,.2f}")

st.dataframe(formatted_summary, use_container_width=True)

st.markdown("---")

# 9. 묶음별 누적 관객수 상위 5개 영화 제목
st.subheader("🏆 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)

for idx, cluster_label in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"#### 묶음 **{cluster_label}**")
        top5_movies = (
            df_clean[df_clean["cluster"] == cluster_label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for rank, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)")
