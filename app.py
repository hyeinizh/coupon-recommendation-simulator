
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. 웹사이트 기본 설정
# ==========================================

st.set_page_config(
    page_title="AI 쿠폰 추천 시뮬레이터",
    page_icon="🎟️",
    layout="wide"
)

st.title("🎟️ 상황 반응형 AI 쿠폰 추천 시뮬레이터")

st.write(
    "개인 특성과 현재 상황을 입력하면 "
    "AI가 쿠폰별 예상 수락확률을 계산합니다."
)

st.info(
    "본 서비스는 설문 기반 머신러닝 예측 시뮬레이터입니다. "
    "실제 구매확률이나 쿠폰 발송 효과를 보장하지 않습니다."
)

# ==========================================
# 2. 학습된 모델 불러오기
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(
        BASE_DIR / "coupon_xgboost_model (2).pkl"
    )
    features = joblib.load(
        BASE_DIR / "coupon_features.pkl"
    )
    return model, features

model, features = load_model()

# ==========================================
# 3. 개인 특성 입력
# ==========================================

st.header("① 개인 특성 입력")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "성별",
        ["Male", "Female"]
    )

    age = st.selectbox(
        "연령",
        ["below21", "21", "26", "31", "36",
         "41", "46", "50plus"]
    )

    income = st.selectbox(
        "연소득 (달러)",
        [
            "Less than $12500",
            "$12500 - $24999",
            "$25000 - $37499",
            "$37500 - $49999",
            "$50000 - $62499",
            "$62500 - $74999",
            "$75000 - $87499",
            "$87500 - $99999",
            "$100000 or More"
        ]
    )

    occupation = st.selectbox(
        "직업",
        [
            "Unemployed",
            "Student",
            "Computer & Mathematical",
            "Education&Training&Library",
            "Office & Administrative Support",
            "Management",
            "Sales & Related",
            "Business & Financial",
            "Healthcare Practitioners & Technical",
            "Healthcare Support",
            "Arts Design Entertainment Sports & Media",
            "Food Preparation & Serving Related",
            "Transportation & Material Moving",
            "Other"
        ]
    )

with col2:
    marital_status = st.selectbox(
        "결혼 상태",
        [
            "Single",
            "Married partner",
            "Unmarried partner",
            "Divorced",
            "Widowed"
        ]
    )

    has_children = st.selectbox(
        "자녀 여부",
        ["없음", "있음"]
    )

    education = st.selectbox(
        "최종 학력",
        [
            "Some college - no degree",
            "Bachelors degree",
            "Graduate degree (Masters or Doctorate)",
            "Associates degree",
            "High School Graduate",
            "Some High School"
        ]
    )

# ==========================================
# 4. 평소 소비습관
# ==========================================

st.header("② 평소 소비습관")

habit_options = [
    "never",
    "less1",
    "1~3",
    "4~8",
    "gt8"
]

h1, h2, h3 = st.columns(3)

with h1:
    bar = st.selectbox("바 방문 빈도 (월)", habit_options)
    coffee = st.selectbox("카페 방문 빈도 (월)", habit_options)

with h2:
    carry = st.selectbox("테이크아웃 이용 빈도 (월)", habit_options)
    restaurant_low = st.selectbox(
        "저가 레스토랑 방문 빈도 (월)",
        habit_options
    )

with h3:
    restaurant_mid = st.selectbox(
        "중가 레스토랑 방문 빈도 (월)",
        habit_options
    )

# ==========================================
# 5. 현재 상황 입력
# ==========================================

st.header("③ 현재 상황 입력")

s1, s2 = st.columns(2)

with s1:
    destination = st.selectbox(
        "이동 목적지",
        ["No Urgent Place", "Home", "Work"]
    )

    passenger = st.selectbox(
        "동행자",
        ["Alone", "Friend(s)", "Partner", "Kid(s)"]
    )

    weather = st.selectbox(
        "날씨",
        ["Sunny", "Rainy", "Snowy"]
    )

    temperature = st.selectbox(
        "기온 (화씨)",
        [30, 55, 80]
    )

with s2:
    time = st.selectbox(
        "현재 시간",
        ["7AM", "10AM", "2PM", "6PM", "10PM"]
    )

    travel_time = st.selectbox(
        "쿠폰 사용 장소까지 이동시간",
        ["15분 미만", "15~25분 미만", "25분 이상"]
    )

    direction = st.selectbox(
        "쿠폰 사용 장소의 방향",
        ["같은 방향", "반대 방향"]
    )

    expiration = st.selectbox(
        "쿠폰 유효기간",
        ["2h", "1d"]
    )

# ==========================================
# 6. 쿠폰 종류
# ==========================================

coupon_types = [
    "Bar",
    "Carry out & Take away",
    "Coffee House",
    "Restaurant(20-50)",
    "Restaurant(<20)"
]

# ==========================================
# 7. 모델 입력값 생성
# ==========================================

def make_input(coupon):

    geq15 = 0
    geq25 = 0

    if travel_time == "15~25분 미만":
        geq15 = 1
    elif travel_time == "25분 이상":
        geq15 = 1
        geq25 = 1

    same = int(direction == "같은 방향")
    opposite = int(direction == "반대 방향")

    data = {
        "destination": destination,
        "passanger": passenger,
        "weather": weather,
        "temperature": temperature,
        "time": time,
        "coupon": coupon,
        "expiration": expiration,
        "gender": gender,
        "age": age,
        "maritalStatus": marital_status,
        "has_children": int(has_children == "있음"),
        "education": education,
        "occupation": occupation,
        "income": income,
        "Bar": bar,
        "CoffeeHouse": coffee,
        "CarryAway": carry,
        "RestaurantLessThan20": restaurant_low,
        "Restaurant20To50": restaurant_mid,
        "toCoupon_GEQ15min": geq15,
        "toCoupon_GEQ25min": geq25,
        "direction_same": same,
        "direction_opp": opposite
    }

    return pd.DataFrame([data])[features]

# ==========================================
# 8. 쿠폰별 예측 및 결과 출력
# ==========================================

st.header("④ AI 쿠폰 추천 결과")

threshold = st.slider(
    "쿠폰 발송 기준 확률 (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5
)

if st.button("🎯 AI 쿠폰 추천 분석하기", type="primary"):

    results = []

    for coupon in coupon_types:
        input_df = make_input(coupon)

        probability = model.predict_proba(
            input_df
        )[0, 1]

        results.append({
            "쿠폰 종류": coupon,
            "예측 수락확률(%)": round(
                probability * 100, 2
            )
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "예측 수락확률(%)",
        ascending=False
    ).reset_index(drop=True)

    st.subheader("쿠폰별 예상 수락확률")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        result_df.set_index("쿠폰 종류")[
            "예측 수락확률(%)"
        ]
    )

    best_coupon = result_df.iloc[0]
    best_prob = best_coupon["예측 수락확률(%)"]

    st.subheader("AI 추천 전략")

    if best_prob >= threshold:
        st.success(
            f"추천 쿠폰: {best_coupon['쿠폰 종류']}\n\n"
            f"예상 수락확률: {best_prob:.2f}%\n\n"
            "추천 전략: 발송"
        )
    else:
        st.warning(
            f"최고 예측 쿠폰: {best_coupon['쿠폰 종류']}\n\n"
            f"예상 수락확률: {best_prob:.2f}%\n\n"
            "추천 전략: 보류"
        )

    st.caption(
        "발송/보류 판단은 사용자가 정한 확률 기준에 "
        "따른 예시 규칙이며, 경제적 최적 전략을 "
        "검증한 결과는 아닙니다."
    )

st.divider()

st.caption(
    "데이터 출처: UCI In-Vehicle Coupon Recommendation | "
    "예측 모델: XGBoost"
)
