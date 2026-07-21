import streamlit as st
import pandas as pd
import joblib
import os

# ── Load artifacts ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model   = joblib.load(os.path.join(BASE_DIR, 'Naive_Loan.pkl'))
scaler  = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title('🏦 Loan Approval Predictor')
st.markdown('Fill in the details below to check loan approval chances.')

# ── Input widgets ────────────────────────────────────────────
age         = st.slider('Person Age', 18, 100, 40)
Sex         = st.selectbox('Gender', ['male', 'female'])
Education   = st.selectbox('Education Level', ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
Income      = st.number_input('Annual Income', 5000, 200000, 12000)
Experience  = st.number_input('Employment Experience (years)', 0, 20, 5)
Home_owner  = st.selectbox('Home Ownership', ['RENT', 'OWN', 'OTHER', 'MORTGAGE'])
Loan        = st.slider('Loan Amount', 100, 50000, 15000)

# ✅ Bug 2 fixed — Yes/No instead of Y/N
PreviousLoan = st.selectbox('Previous Loan Default', ['Yes', 'No'])

Interest    = st.slider('Loan Interest Rate (%)', 0.0, 20.0, 4.0)

# ✅ Bug 3 fixed — removed DEBTCONSOLIDATION
Intention   = st.selectbox('Loan Intent', ['MEDICAL', 'PERSONAL', 'EDUCATION', 'VENTURE', 'HOMEIMPROVEMENT'])

Per_income  = st.slider('Loan as % of Income', 0.0, 0.9, 0.3)

# ✅ Bug 1 fixed — cb_ not b_
Credit_hist  = st.slider('Credit History Length (years)', 0.0, 20.0, 4.0)
Credit_score = st.slider('Credit Score', 100.0, 1000.0, 200.0)

# ── Prediction ───────────────────────────────────────────────
if st.button('🔍 Predict Loan Approval'):

    raw_input = {
        'person_age'                                   : age,
        'person_income'                                : Income,
        'person_emp_exp'                               : Experience,
        'loan_amnt'                                    : Loan,
        'loan_int_rate'                                : Interest,
        'loan_percent_income'                          : Per_income,
        'cb_person_cred_hist_length'                   : Credit_hist,
        'credit_score'                                 : Credit_score,
        'person_gender_'             + Sex             : 1,
        'person_education_'          + Education       : 1,
        'person_home_ownership_'     + Home_owner      : 1,
        'previous_loan_defaults_on_file_' + PreviousLoan : 1,
        'loan_intent_'               + Intention       : 1,
    }

    # Build DataFrame and fill missing one-hot columns with 0
    input_df = pd.DataFrame([raw_input])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    # ✅ THE FIX — scale only the numeric columns the scaler knows
    numeric_cols = scaler.feature_names_in_.tolist()  # reads exactly what scaler was trained on
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])  # scale in place

    # Now pass full input_df to the model (already has correct columns)
    prediction = model.predict(input_df)
    proba      = model.predict_proba(input_df)[0]
    confidence = round(max(proba) * 100)

    if prediction[0] == 1:
        st.success(f'✅ Loan Approved — {confidence}% confidence')
    else:
        st.error(f'❌ Loan Not Approved — {confidence}% confidence')