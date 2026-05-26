import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------
# LOAD MODEL
# -------------------------
import os
import pickle

BASE_DIR = os.path.dirname(__file__)

pipe_path = os.path.join(BASE_DIR, 'pipe.pkl')
df_path = os.path.join(BASE_DIR, 'df.pkl')

pipe = pickle.load(open(pipe_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

st.title("💻 Laptop Price Predictor (DEBUG MODE)")

# -------------------------
# NORMALIZE COLUMN NAMES (IMPORTANT FIX)
# -------------------------
df.columns = df.columns.str.strip()

# -------------------------
# GET EXPECTED FEATURES
# -------------------------
try:
    expected_features = pipe.named_steps['columntransformer'].feature_names_in_
except Exception:
    expected_features = None

# -------------------------
# INPUTS
# -------------------------
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of Laptop (kg)', 0.5, 5.0, 2.0)

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])

screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# -------------------------
# PREDICTION BUTTON
# -------------------------
if st.button('Predict Price'):

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # -------------------------
    # BUILD QUERY DATAFRAME
    # -------------------------
    query = pd.DataFrame([[
        company,
        type_name,
        ram,
        weight,
        touchscreen_val,
        ips_val,
        ppi,
        cpu,
        hdd,
        ssd,
        gpu,
        os
    ]], columns=[
        'Company',
        'TypeName',
        'Ram',
        'Weight',
        'Touchscreen',
        'Ips',
        'ppi',   # ✅ FIXED (was Ppi earlier)
        'Cpu brand',
        'HDD',
        'SSD',
        'Gpu brand',
        'os'
    ])

    st.subheader("📌 Input DataFrame")
    st.dataframe(query)

    # -------------------------
    # NORMALIZE QUERY COLUMNS (CRITICAL FIX)
    # -------------------------
    query.columns = query.columns.str.strip()

    # -------------------------
    # DEBUG FEATURE CHECK
    # -------------------------
    if expected_features is not None:
        st.subheader("🔍 Feature Validation")

        query_cols = set(query.columns)
        model_cols = set(expected_features)

        missing = model_cols - query_cols
        extra = query_cols - model_cols

        st.write("Model expects:", list(model_cols))
        st.write("You provided:", list(query_cols))

        if missing:
            st.error(f"❌ Missing columns: {missing}")

        if extra:
            st.warning(f"⚠️ Extra columns: {extra}")

        if not missing and not extra:
            st.success("✅ Perfect feature match!")

    # -------------------------
    # CLEAN OBJECT COLUMNS
    # -------------------------
    for col in query.columns:
        if query[col].dtype == "object":
            query[col] = query[col].astype(str).str.strip()

    # -------------------------
    # PREDICTION
    # -------------------------
    try:
        st.subheader("🚀 Prediction Debug")

        st.write("Final model input:")
        st.dataframe(query)

        prediction = pipe.predict(query)[0]
        price = int(np.exp(prediction))  # reverse log transform

        st.success(f"💰 Predicted Price: ₹ {price:,}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

        st.subheader("🧠 Debug Info")
        st.write("Expected features:")
        st.write(expected_features)
