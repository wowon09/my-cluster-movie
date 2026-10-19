import math
import random
import pandas as pd
import streamlit as st

# 1. 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

# 맨 위 제목 및 아이콘
st.title("🎬 영화 유형 나누기")

# 데이터 로드
DATA_URL = (
    "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

total_raw_count = len(raw_df)

# 전처리: 속성 값 미존재 또는 first_week_audi == 0 제외
required_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df_clean = raw_df.dropna(subset=required_cols).copy()
df_clean = df_clean[df_clean["first_week_audi"] > 0].copy()

# 파생 속성 생성 (상용로그, 롱런지수 20 클리핑)
df_clean["log_scrn"] = df_clean["first_scrn"].apply(
    lambda x: math.log10(x) if x > 0 else 0
)
df_clean["log_total_audi"] = df_clean["total_audi"].apply(
    lambda x: math.log10(x) if x > 0 else 0
)
df_clean["long_run"] = (df_clean["total_audi"] / df_clean["first_week_audi"]).apply(
    lambda x: min(x, 20.0)
)

used_count = len(df_clean)

# 전체 편수와 묶은 편수를 화면에 한 줄로 표시
st.write(
    f"**전체 영화 편수:** {total_raw_count}편 | **군집 분석 대상 편수:** {used_count}편"
)
st.markdown("---")

# 속성 매핑 사전
feature_dict = {
    "스크린 수 (상용로그)": "log_scrn",
    "누적 관객 (상용로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run",
}

# 속성 선택 UI (기본은 4개 모두 선택)
selected_feature_names = st.multiselect(
    "군집화(묶기)에 사용할 속성을 선택하세요 (둘 이상 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_names) < 2:
    st.warning(
        "⚠️ 속성을 둘 이상 고르셔야 군집화를 진행할 수 있습니다. 2개 이상 선택해 주세요."
    )
    st.stop()

selected_cols = [feature_dict[name] for name in selected_feature_names]

# --- Pure Python / Pandas K-Means 구현 ---
# 표준화 (Z-Score)
X_data = df_clean[selected_cols].values.tolist()
num_features = len(selected_cols)
num_samples = len(X_data)

# 각 속성별 평균 및 표준편차 계산
means = []
stds = []
for j in range(num_features):
    col_vals = [X_data[i][j] for i in range(num_samples)]
    m = sum(col_vals) / num_samples
    variance = sum((x - m) ** 2 for x in col_vals) / num_samples
    s = math.sqrt(variance) if variance > 0 else 1.0
    means.append(m)
    stds.append(s)

# 표준화된 데이터
X_scaled = []
for i in range(num_samples):
    row = [(X_data[i][j] - means[j]) / stds[j] for j in range(num_features)]
    X_scaled.append(row)

# K-Means (K=3, random_state=42 난수 고정)
random.seed(42)
K = 3

# 초기 중심점 3개 랜덤 선택
init_indices = random.sample(range(num_samples), K)
centroids = [list(X_scaled[idx]) for idx in init_indices]

labels = [0] * num_samples
for _ in range(50):
    # 각 데이터와 가장 가까운 중심점 찾기
    new_labels = []
    for i in range(num_samples):
        min_dist = float("inf")
        closest_k = 0
        for k in range(K):
            dist = sum(
                (X_scaled[i][j] - centroids[k][j]) ** 2 for j in range(num_features)
            )
            if dist < min_dist:
                min_dist = dist
                closest_k = k
        new_labels.append(closest_k)

    if new_labels == labels and _ > 0:
        break
    labels = new_labels

    # 중심점 업데이트
    for k in range(K):
        cluster_points = [
            X_scaled[i] for i in range(num_samples) if labels[i] == k
        ]
        if cluster_points:
            for j in range(num_features):
                centroids[k][j] = sum(p[j] for p in cluster_points) / len(
                    cluster_points
                )

df_clean["cluster_temp"] = labels

# 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 매핑
cluster_audi_mean = df_clean.groupby("cluster_temp")["total_audi"].mean()
sorted_clusters = cluster_audi_mean.sort_values(ascending=False).index.tolist()

label_map = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰",
}
df_clean["묶음"] = df_clean["cluster_temp"].map(label_map)

st.markdown("---")

# 2차원 산점도
st.subheader("📊 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_2d_name = st.selectbox(
        "가로축 속성 선택:", options=selected_feature_names, index=0, key="x2d"
    )
with col2d_2:
    y_2d_name = st.selectbox(
        "세로축 속성 선택:",
        options=selected_feature_names,
        index=min(1, len(selected_feature_names) - 1),
        key="y2d",
    )

x_col = feature_dict[x_2d_name]
y_col = feature_dict[y_2d_name]

# Streamlit 내장 차트 (마우스 오버 시 영화 제목 표시 지원)
st.scatter_chart(
    df_clean,
    x=x_col,
    y=y_col,
    color="묶음",
    size=50,
)

st.markdown("---")

# 3차원 안내 및 축 선택
st.subheader("🎲 3차원 속성 설정")
if len(selected_feature_names) < 3:
    st.info("💡 고른 속성이 셋보다 적어 3차원 축 설정 대신 안내 문구를 표시합니다. 속성을 3개 이상 골라주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        st.selectbox("X축 속성:", options=selected_feature_names, index=0, key="x3d")
    with col3d_2:
        st.selectbox(
            "Y축 속성:",
            options=selected_feature_names,
            index=min(1, len(selected_feature_names) - 1),
            key="y3d",
        )
    with col3d_3:
        st.selectbox(
            "Z축 속성:",
            options=selected_feature_names,
            index=min(2, len(selected_feature_names) - 1),
            key="z3d",
        )
    st.caption("※ Streamlit 및 Pandas 기본 전용 모드에서는 2차원 산점도로 시각화가 지원됩니다.")

st.markdown("---")

# 묶음별 편수 및 네 속성의 평균(원래 단위) 표
st.subheader("📋 묶음별 요약 통계 (원래 단위 평균)")

summary_df = (
    df_clean.groupby("묶음")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        top10일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

# 표 포맷 지정
summary_formatted = summary_df.copy()
summary_formatted["편수"] = summary_formatted["편수"].map("{:,}편".format)
summary_formatted["스크린수_평균"] = summary_formatted["스크린수_평균"].map(
    "{:,.1f}".format
)
summary_formatted["누적관객_평균"] = summary_formatted["누적관객_평균"].map(
    "{:,.0f}".format
)
summary_formatted["top10일수_평균"] = summary_formatted["top10일수_평균"].map(
    "{:,.1f}".format
)
summary_formatted["롱런지수_평균"] = summary_formatted["롱런지수_평균"].map(
    "{:,.2f}".format
)

st.dataframe(summary_formatted, use_container_width=True)

st.markdown("---")

# 묶음마다 누적 관객이 많은 순서로 영화 5편의 제목 표시
st.subheader("🏆 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)

for idx, cluster_label in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"#### 묶음 **{cluster_label}**")
        top5_movies = (
            df_clean[df_clean["묶음"] == cluster_label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for rank, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)")
            
