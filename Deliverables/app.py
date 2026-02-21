import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_data
def load_and_train():
    df = pd.read_csv(r'C:\Users\admin\Desktop\project-team-x\bank.csv')
    df['deposit_bin'] = (df['deposit'] == 'yes').astype(int)

    df_encoded = df.copy()
    le_dict = {}
    cat_cols = ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'poutcome']

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le

    df_encoded.drop(columns=['deposit'], inplace=True)
    features = [c for c in df_encoded.columns if c != 'deposit_bin']

    X = df_encoded[features]
    y = df_encoded['deposit_bin']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_rfe = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator=lr_rfe, n_features_to_select=8)
    rfe.fit(X_scaled, y)

    selected_features = [f for f, s in zip(features, rfe.support_) if s]
    X_selected = df_encoded[selected_features]
    X_scaled_selected = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, le_dict, selected_features, df

model, scaler, le_dict, selected_features, df = load_and_train()

st.title("Bank Term Deposit Subscription Predictor")
st.markdown("**Team Fintech** | DATA 200 – Applied Statistical Analysis")
st.markdown("---")
st.markdown("Enter client information below to predict the likelihood of subscribing to a term deposit.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=95, value=35)
    job = st.selectbox("Job", sorted(df['job'].unique()))
    marital = st.selectbox("Marital Status", sorted(df['marital'].unique()))
    education = st.selectbox("Education", sorted(df['education'].unique()))
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=1000)
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])

with col2:
    contact = st.selectbox("Contact Type", sorted(df['contact'].unique()))
    day = st.number_input("Day of Month Contacted", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month of Contact", ['jan','feb','mar','apr','may','jun',
                                               'jul','aug','sep','oct','nov','dec'])
    duration = st.number_input("Last Call Duration (seconds)", min_value=0, max_value=5000, value=300)
    campaign = st.number_input("Number of Contacts This Campaign", min_value=1, max_value=50, value=2)
    pdays = st.number_input("Days Since Last Contact (-1 = never)", min_value=-1, max_value=1000, value=-1)
    previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=60, value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", sorted(df['poutcome'].unique()))

st.markdown("---")

if st.button("Predict Subscription Likelihood", use_container_width=True):
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }

    input_df = pd.DataFrame([input_data])

    for col in ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'poutcome']:
        if col in le_dict:
            try:
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0

    available_features = [f for f in selected_features if f in input_df.columns]
    input_selected = input_df[available_features]
    input_scaled = scaler.transform(input_selected)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    prob_yes = probability[1]
    prob_no  = probability[0]

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.success(f"This client is **likely to subscribe** to a term deposit.")
    else:
        st.error(f"This client is **unlikely to subscribe** to a term deposit.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probability of Subscribing", f"{prob_yes*100:.2f}%")
    with col_b:
        st.metric("Probability of Not Subscribing", f"{prob_no*100:.2f}%")

    st.markdown("#### Confidence Level")
    st.progress(float(prob_yes))

    st.markdown("---")
    st.markdown("#### Key Factors for This Prediction")

    coef_df = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)

    top_positive = coef_df[coef_df['Coefficient'] > 0].head(3)
    top_negative = coef_df[coef_df['Coefficient'] < 0].head(3)

    factor_col1, factor_col2 = st.columns(2)
    with factor_col1:
        st.markdown("**Factors increasing likelihood:**")
        for _, row in top_positive.iterrows():
            st.markdown(f"- {row['Feature']} (+{row['Coefficient']:.3f})")
    with factor_col2:
        st.markdown("**Factors decreasing likelihood:**")
        for _, row in top_negative.iterrows():
            st.markdown(f"- {row['Feature']} ({row['Coefficient']:.3f})")

st.markdown("---")
st.markdown(
    "<small>DATA 200 Project | Team Fintech | Saugat Ojha · Rakesh Kumar Sah · Naitik Shrestha</small>",
    unsafe_allow_html=True
)
