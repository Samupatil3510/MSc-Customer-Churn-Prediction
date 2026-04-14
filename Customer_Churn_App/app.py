import streamlit as st
import pandas as pd
import joblib
from reportlab.pdfgen import canvas
import logging
import os
import datetime
import time
import random
import matplotlib.pyplot as plt

# -----------------------------
# LOGGING
# -----------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# SESSION
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(datetime.datetime.now())

if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ IMPORTANT (fix feature mismatch)
feature_names = scaler.feature_names_in_

# -----------------------------
# FUNCTION
# -----------------------------
def predict_churn(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    return prediction, probability

# -----------------------------
# PDF FUNCTION
# -----------------------------
def create_pdf(tenure, monthly, total, contract, result, probability):
    file_path = "customer_churn_report.pdf"

    c = canvas.Canvas(file_path)
    c.setFont("Helvetica", 12)

    c.drawString(100, 800, "Customer Churn Prediction Report")
    c.drawString(100, 760, f"Tenure: {tenure}")
    c.drawString(100, 740, f"Monthly Charges: {monthly}")
    c.drawString(100, 720, f"Total Charges: {total}")
    c.drawString(100, 700, f"Contract: {contract}")
    c.drawString(100, 660, f"Prediction: {result}")
    c.drawString(100, 640, f"Probability: {round(probability*100,2)}%")

    c.save()
    return file_path

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Dashboard")
st.sidebar.write("Customer Churn Prediction")
st.sidebar.write("Session:", st.session_state.session_id)

# -----------------------------
# UI
# -----------------------------
st.title("Customer Churn Prediction System")

form_key = st.session_state.form_key

tenure = st.number_input("Tenure", 0, 100, 0, key=f"tenure_{form_key}")
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 0.0, key=f"monthly_{form_key}")
total = st.number_input("Total Charges", 0.0, 10000.0, 0.0, key=f"total_{form_key}")

senior = st.selectbox("Senior Citizen", ["No", "Yes"], key=f"senior_{form_key}")
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"], key=f"contract_{form_key}")

colA, colB, colC = st.columns(3)

with colA:
    predict_button = st.button("Predict")

with colB:
    clear_button = st.button("Clear")

with colC:
    random_button = st.button("Random Data")

# -----------------------------
# CLEAR
# -----------------------------
if clear_button:
    st.session_state.form_key += 1
    st.session_state.pop("result", None)
    st.session_state.pop("probability", None)
    st.rerun()

# -----------------------------
# RANDOM DATA
# -----------------------------
if random_button:
    st.session_state.form_key += 1
    fk = st.session_state.form_key

    st.session_state[f"tenure_{fk}"] = random.randint(1, 60)
    st.session_state[f"monthly_{fk}"] = random.uniform(20, 120)
    st.session_state[f"total_{fk}"] = random.uniform(100, 5000)
    st.session_state[f"senior_{fk}"] = random.choice(["Yes", "No"])
    st.session_state[f"contract_{fk}"] = random.choice(["Month-to-month","One year","Two year"])

    st.rerun()

# -----------------------------
# PREDICT
# -----------------------------
if predict_button:

    try:
        start = time.time()

        senior_val = 1 if senior == "Yes" else 0
        contract_one = 1 if contract == "One year" else 0
        contract_two = 1 if contract == "Two year" else 0

        # create full feature set
        data_dict = {col: [0] for col in feature_names}

        if 'SeniorCitizen' in data_dict:
            data_dict['SeniorCitizen'] = [senior_val]

        if 'tenure' in data_dict:
            data_dict['tenure'] = [tenure]

        if 'MonthlyCharges' in data_dict:
            data_dict['MonthlyCharges'] = [monthly]

        if 'TotalCharges' in data_dict:
            data_dict['TotalCharges'] = [total]

        if 'Contract_One year' in data_dict:
            data_dict['Contract_One year'] = [contract_one]

        if 'Contract_Two year' in data_dict:
            data_dict['Contract_Two year'] = [contract_two]

        # smart defaults
        for col in data_dict:
            if "PhoneService_Yes" in col:
                data_dict[col] = [1]
            if "Partner_Yes" in col:
                data_dict[col] = [1]

        data = pd.DataFrame(data_dict)
        data = data[feature_names]

        prediction, probability = predict_churn(data)

        end = time.time()

        logging.info(f"Prediction: {prediction}, Probability: {probability}")

        # KPI
        st.subheader("KPI")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Probability", f"{round(probability*100,2)}%")

        with c2:
            st.metric("Status", "Churn" if prediction==1 else "Safe")

        result = "CHURN ❌" if prediction==1 else "STAY ✅"

        st.write(result)
        st.progress(float(probability))
        st.write("Time:", round(end-start,4), "seconds")

        st.session_state.result = result
        st.session_state.probability = probability

        # chart
        st.subheader("Visualization")
        fig, ax = plt.subplots()
        ax.bar(["Stay","Churn"], [1-probability, probability])
        st.pyplot(fig)

    except Exception as e:
        logging.error(str(e))
        st.error(f"Error: {str(e)}")

# -----------------------------
# PDF DOWNLOAD
# -----------------------------
if "result" in st.session_state:

    pdf_path = create_pdf(
        tenure,
        monthly,
        total,
        contract,
        st.session_state.result,
        st.session_state.probability
    )

    with open(pdf_path, "rb") as file:
        st.download_button(
            label="Download PDF Report",
            data=file.read(),
            file_name="Customer_Churn_Report.pdf",
            mime="application/pdf"
        )

    st.success("PDF saved in app folder ✅")

# -----------------------------
# EXPLANATION
# -----------------------------
if "probability" in st.session_state:

    p = st.session_state.probability

    st.subheader("Explanation")

    if p < 0.3:
        st.write("Low risk")
    elif p < 0.6:
        st.write("Medium risk")
    else:
        st.write("High risk")