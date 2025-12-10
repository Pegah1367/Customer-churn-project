import streamlit as st
import numpy as np
import joblib

# ----------------- Config -----------------
LOW_THR = 0.30
HIGH_THR = 0.70

# ----------------- Load models -----------------
@st.cache_resource
def load_models():
    loaded = joblib.load("telco_churn_models.pkl")
    xgb_teacher = loaded["xgb_teacher"]
    xgb_risk = loaded["xgb_risk"]
    feature_names = loaded["feature_names"]
    return xgb_teacher, xgb_risk, feature_names

xgb_teacher, xgb_risk, feature_names = load_models()

# ----------------- Feature builder -----------------
def build_X_from_input(input_dict, feature_names):
    """
    input_dict: مقادیر فرم
    feature_names: لیست 1169 فیچر که همراه مدل ذخیره شده
    """
    X = np.zeros((1, len(feature_names)))

    def set_val(feat, val):
        if feat in feature_names:
            X[0, feature_names.index(feat)] = val

    # --- numeric features ---
    set_val("Tenure in Months", input_dict["tenure"])
    set_val("Monthly Charge", input_dict["monthly_charge"])
    set_val("Total Charges", input_dict["total_charges"])
    set_val("Avg Monthly Long Distance Charges", input_dict["avg_long_distance"])
    set_val("Number of Referrals", input_dict["num_referrals"])
    set_val("Satisfaction Score", input_dict["satisfaction"])

    # --- categorical (one-hot) ---
    set_val(f"Internet Type_{input_dict['internet_type']}", 1)
    set_val(f"Contract_{input_dict['contract']}", 1)

    return X

# ----------------- Business recommendations -----------------
def get_business_recommendations(risk_level, info):
    recs = []

    if risk_level == "High":
        recs.append("High-risk customer: immediate retention actions recommended.")
        recs.append("Offer a strong incentive (e.g., 15–20% discount + free upgrade) for 1-year commitment.")
        recs.append("Have an agent contact the customer within 24–48 hours.")
    elif risk_level == "Medium":
        recs.append("Moderate churn risk: proactive engagement recommended.")
        recs.append("Send a personalized offer (small discount or loyalty points) and survey about service quality.")
    else:  # Low
        recs.append("Low churn risk: keep customer engaged with regular loyalty rewards.")
        recs.append("Promote cross-sell/upsell opportunities based on usage.")

    # مثال ساده با Satisfaction
    if info["satisfaction"] <= 2:
        recs.append("Customer satisfaction is low (1–2/5): prioritize service quality review and support follow-up.")

    return recs

# ----------------- Streamlit UI -----------------
def main():
    st.set_page_config(
        page_title="Customer Churn Prediction – Offline Demo",
        page_icon="📡",
        layout="centered"
    )

    st.title("Customer Churn Prediction")
    st.caption("Offline demo using your local XGBoost models")

    st.markdown("---")
    st.subheader("Customer Information")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=1, step=1)
            monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, value=85.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=85.0, step=1.0)
            num_referrals = st.number_input("Number of Referrals", min_value=0, value=0, step=1)

        with col2:
            satisfaction = st.slider("Satisfaction Score (1-5)", min_value=1, max_value=5, value=1)
            internet_type = st.selectbox(
                "Internet Type",
                ["Fiber Optic", "Cable", "DSL", "None"]  # این را با مقادیر واقعی دیتاست خودت هماهنگ کن
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-Month", "One Year", "Two Year"]
            )
            avg_long_distance = st.number_input(
                "Avg Long Distance Charges ($)",
                min_value=0.0,
                value=22.0,
                step=1.0
            )

        submit = st.form_submit_button("Predict Churn")

    if submit:
        user_input = {
            "tenure": tenure,
            "monthly_charge": monthly_charge,
            "total_charges": total_charges,
            "avg_long_distance": avg_long_distance,
            "num_referrals": num_referrals,
            "satisfaction": satisfaction,
            "internet_type": internet_type,
            "contract": contract,
        }

        X = build_X_from_input(user_input, feature_names)

        # Sanity check
        st.write(f"Non-zero features: {np.count_nonzero(X)}")

        # Teacher model (binary churn)
        proba = float(xgb_teacher.predict_proba(X)[0, 1])
        teacher_pred = int(proba >= 0.5)

        # Risk buckets
        if proba < LOW_THR:
            risk = "Low"
        elif proba > HIGH_THR:
            risk = "High"
        else:
            risk = "Medium"

        # --- Display results ---
        st.markdown("---")
        st.subheader("Prediction Results")

        st.write(f"**Teacher model prediction (0 = No churn, 1 = Churn):** {teacher_pred}")
        st.write(f"**Churn probability:** {proba:.4f}")
        st.write(f"**Risk level:** {risk}")

        st.progress(proba if 0.0 <= proba <= 1.0 else 0.0)

        st.markdown("### Business Recommendations")
        recs = get_business_recommendations(risk, user_input)
        for r in recs:
            st.write(f"- {r}")

if __name__ == "__main__":
    main()
