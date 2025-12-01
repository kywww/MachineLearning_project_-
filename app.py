import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  # Windows용
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

# ------------------------------
# 1️⃣ 캐시된 함수
# ------------------------------
@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

@st.cache_resource
def load_xgb_model(path):
    model = xgb.Booster()
    model.load_model(path)
    return model

@st.cache_resource
def load_lgbm_model(path):
    return joblib.load(path)

@st.cache_resource
def load_joblib_model(path):
    return joblib.load(path)

@st.cache_resource
def load_encoders(path):
    return joblib.load(path)

# ------------------------------
# 2️⃣ 파일 경로 / 파일명
# ------------------------------
EXCEL_PATH = "2total_daily_data.xlsx"

# XGBoost
XGB_MODEL_PATH = "xgboost_model.json"
XGB_ENCODER_PATH = "label_encoders.joblib"

# Linear Regression
LINEAR_MODEL_PATH = "linear_model.pkl"
LINEAR_COLS_PATH = "linear_model_columns.pkl"
LINEAR_ENCODER_PATH = XGB_ENCODER_PATH  # 기존 XGB 인코더 사용

# Random Forest
RF_MODEL_PATH = "rf_model_compressed.pkl"
RF_ENCODER_PATH = "rf_label_encoders.pkl"

# LightGBM
LGBM_MODEL_PATH = "future_lgbm_model.pkl"
LGBM_ENCODER_PATH = "future_label_encoders.pkl"

# ------------------------------
# 3️⃣ 데이터/모델 불러오기
# ------------------------------
df = load_excel(EXCEL_PATH)

xgb_model = load_xgb_model(XGB_MODEL_PATH)
xgb_encoders = load_encoders(XGB_ENCODER_PATH)

linear_model = load_joblib_model(LINEAR_MODEL_PATH)
linear_cols = load_joblib_model(LINEAR_COLS_PATH)
linear_encoders = load_encoders(LINEAR_ENCODER_PATH)

rf_model = load_joblib_model(RF_MODEL_PATH)
rf_encoders = load_encoders(RF_ENCODER_PATH)

lgbm_model = load_lgbm_model(LGBM_MODEL_PATH)
lgbm_encoders = load_encoders(LGBM_ENCODER_PATH)

# 모델 딕셔너리
models_dict = {
    "XGBoost": xgb_model,
    "Linear Regression": linear_model,
    "Random Forest": rf_model,
    "LightGBM": lgbm_model
}

# ------------------------------
# 4️⃣ Streamlit UI
# ------------------------------
st.title("📦 편의점 수요 예측 시스템")

# 모델 선택
model_name = st.selectbox("모델 선택", list(models_dict.keys()))

# 요일 선택 → 주말 여부 자동 계산
요일_list = ["월","화","수","목","금","토","일"]
selected_day = st.selectbox("요일 선택", 요일_list)
주말여부 = 1 if selected_day in ["토","일"] else 0
st.text(f"주말 여부: {'주말' if 주말여부==1 else '평일'}")

# 지역 선택
regions = df['지역'].unique().tolist()
selected_region = st.selectbox("지역 선택", regions)

# 선택한 지역 매장 선택
region_stores = df[df['지역']==selected_region]['매장'].unique().tolist()
selected_store = st.selectbox("매장 선택", region_stores)

# 물품 선택
items = df['물품'].unique().tolist()
selected_item = st.selectbox("물품 선택", items)

# 온도 선택
temp_list = ["-10~0°C","1~10°C","11~20°C","21~30°C","31~40°C"]
selected_temp = st.selectbox("평균 기온 구간", temp_list)
temp_mapping = {"-10~0°C":0,"1~10°C":5,"11~20°C":15,"21~30°C":25,"31~40°C":35}
온도_val = temp_mapping[selected_temp]

# 날씨 선택
weather_list = df['날씨'].unique().tolist()
selected_weather = st.selectbox("날씨 선택", weather_list)

# ------------------------------
# 5️⃣ 안전한 transform 함수
# ------------------------------
def safe_transform(encoder, values):
    known = set(encoder.classes_)
    result = []
    for v in values:
        if v in known:
            result.append(encoder.transform([v])[0])
        else:
            result.append(encoder.transform([encoder.classes_[0]])[0])
    return result

# ------------------------------
# 6️⃣ 예측 버튼
# ------------------------------
if st.button("예상 수요 확인"):
    # 입력 데이터 생성
    input_df = pd.DataFrame({
        "요일":[selected_day],
        "주말여부":[주말여부],
        "지역":[selected_region],
        "매장":[selected_store],
        "물품":[selected_item],
        "온도":[온도_val],
        "날씨":[selected_weather]
    })

    # ------------------------------
    # 모델별 예측
    # ------------------------------
    if model_name == "XGBoost":
        encode_cols = ["요일","지역","매장","물품","날씨"]
        for col in encode_cols:
            encoder = xgb_encoders[col]
            input_df[col] = safe_transform(encoder, input_df[col].astype(str))
        dmatrix = xgb.DMatrix(input_df)
        pred = models_dict[model_name].predict(dmatrix)[0]

    elif model_name == "Linear Regression":
        input_df = input_df.reindex(columns=linear_cols, fill_value=0)
        pred = models_dict[model_name].predict(input_df)[0]

    elif model_name == "Random Forest":
        encode_cols = ["요일","지역","매장","물품","날씨"]
        for col in encode_cols:
            encoder = rf_encoders[col]
            input_df[col] = safe_transform(encoder, input_df[col].astype(str))
        pred = models_dict[model_name].predict(input_df)[0]

    elif model_name == "LightGBM":
        encode_cols = ["요일","지역","매장","물품","날씨"]
        for col in encode_cols:
            encoder = lgbm_encoders[col]
            input_df[col] = safe_transform(encoder, input_df[col].astype(str))
        pred = models_dict[model_name].predict(input_df)[0]

    예측수요 = np.round(pred)
    권장발주량 = np.round(pred * 1.1)

    # ------------------------------
    # 7️⃣ 결과 출력
    # ------------------------------
    st.subheader("📊 예측 결과")
    st.write(f"선택 모델: {model_name}")
    st.write(f"예측 수요: {예측수요}")
    st.write(f"권장 발주량 (10% 여유): {권장발주량}")

    # ------------------------------
    # 8️⃣ 과거 조건과 비교 그래프
    # ------------------------------
    filtered_df = df[
        (df['요일'] == selected_day) &
        (df['지역'] == selected_region) &
        (df['매장'] == selected_store) &
        (df['물품'] == selected_item) &
        (df['날씨'] == selected_weather) &
        (df['온도'] >= 온도_val - 5) & (df['온도'] <= 온도_val + 5)
    ]

    if len(filtered_df) > 0:
        plt.figure(figsize=(10,5))
        plt.bar(range(len(filtered_df)), filtered_df['수요물품수'], color='skyblue', label='과거 판매량')
        plt.axhline(예측수요, color='red', linestyle='--', label='예측 수요')
        plt.xlabel("과거 샘플")
        plt.ylabel("판매량 (수요물품수)")
        plt.title("선택 조건 기준 과거 판매량 vs 예측 수요")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("선택 조건에 해당하는 과거 데이터가 없습니다.")

    # ------------------------------
    # 9️⃣ 동적 이유 생성
    # ------------------------------
    reasons = []
    if 주말여부 == 1:
        reasons.append(f"- {selected_day}요일은 주말이라 일반적으로 수요가 더 높습니다.")
    else:
        reasons.append(f"- {selected_day}요일은 평일이라 평균적인 수요 수준입니다.")
    reasons.append(f"- 선택된 지역/매장({selected_region} / {selected_store})의 과거 판매 패턴을 반영했습니다.")
    reasons.append(f"- 물품({selected_item})은 과거 데이터 기준으로 평균 판매량이 계산되었습니다.")
    reasons.append(f"- 날씨({selected_weather})와 온도({selected_temp})에 따라 구매량에 영향을 줄 수 있습니다.")
    reasons.append("→ 모델은 입력 조건과 가장 유사한 과거 데이터를 기반으로 예측했습니다.")
    st.info("\n".join(reasons))
