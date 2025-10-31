# Streamlit App: Prediksi Permintaan Taksi NYC
# ==========================================================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. Konfigurasi Halaman
# ----------------------------------------------------------
st.set_page_config(page_title="Prediksi Permintaan Taksi NYC", page_icon="🚕", layout="wide")

st.title("🚖 Prediksi Permintaan Taksi NYC")
st.markdown("""
Aplikasi ini memprediksi **jumlah trip taksi** berdasarkan:
- Lokasi penjemputan (PULocationID)
- Jam penjemputan
- Hari dalam seminggu
""")

st.divider()

# ----------------------------------------------------------
# 2. Load Model & Data
# ----------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_model.pkl")
    df = pd.read_csv("taxi_tripdata.csv")
    return model, df

model, df = load_model_and_data()

# ----------------------------------------------------------
# 3. Input Parameter
# ----------------------------------------------------------
st.subheader("Masukkan Parameter Prediksi")

col1, col2, col3 = st.columns(3)
with col1:
    pulocationid = st.number_input("PULocationID", min_value=int(df["pulocationid"].min()),
                                   max_value=int(df["pulocationid"].max()), value=int(df["pulocationid"].mode()[0]))
with col2:
    pickup_hour = st.slider("Jam Penjemputan", 0, 23, 12)
with col3:
    pickup_dayofweek = st.slider("Hari Penjemputan (0=Senin, 6=Minggu)", 0, 6, 2)

# ----------------------------------------------------------
# 4. Prediksi
# ----------------------------------------------------------
if st.button("🔍 Prediksi Jumlah Trip"):
    input_data = pd.DataFrame({
        "pulocationid": [pulocationid],
        "pickup_hour": [pickup_hour],
        "pickup_dayofweek": [pickup_dayofweek]
    })

    pred = model.predict(input_data)[0]
    st.success(f"🚕 **Prediksi Jumlah Trip:** {pred:.2f} perjalanan")

# ----------------------------------------------------------
# 5. Analisis Data (EDA)
# ----------------------------------------------------------
st.divider()
st.subheader("📊 Analisis Data (EDA)")

if st.checkbox("Tampilkan Distribusi Trip per Hari"):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x="pickup_dayofweek", data=df, color="orange")
    ax.set_title("Distribusi Trip Taksi per Hari (0=Senin ... 6=Minggu)")
    ax.set_xlabel("Hari ke- (0–6)")
    ax.set_ylabel("Jumlah Trip")
    st.pyplot(fig)

if st.checkbox("Tampilkan Distribusi Trip per Jam"):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x="pickup_hour", data=df, color="skyblue")
    ax.set_title("Distribusi Trip Taksi per Jam (0–23)")
    ax.set_xlabel("Jam Penjemputan")
    ax.set_ylabel("Jumlah Trip")
    st.pyplot(fig)

st.caption("© 2025 – Prediksi Permintaan Taksi NYC | Dibuat oleh Hazril")
