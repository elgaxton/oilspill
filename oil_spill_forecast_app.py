import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Oil Spill Forecasting App", layout="wide")
st.title("🛢️ Oil Spill Forecasting Dashboard")

# Zonal office name mapping
zonal_office_names = {
   'ak': 'Akure',
   'by': 'Bayelsa',
   'kd': 'Kaduna',
   'lg': 'Lagos',
   'ph': 'Port Harcourt',
   'uy': 'Uyo',
   'wa': 'Warri'
}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("nosdra_cleaned.csv")
    df['incidentdate'] = pd.to_datetime(df['incidentdate'], errors='coerce')
    df['estimatedquantity'] = pd.to_numeric(df['estimatedquantity'], errors='coerce')

    if 'location' in df.columns:
        df['location'] = df['location'].astype(str)
        df = df.dropna(subset=['incidentdate', 'estimatedquantity', 'location'])
    else:
        df = df.dropna(subset=['incidentdate', 'estimatedquantity'])

    df = df[df['estimatedquantity'] > 0]

    # Apply zonaloffice mapping if column exists
    if 'zonaloffice' in df.columns:
        df['zonaloffice'] = df['zonaloffice'].str.lower().map(zonal_office_names).fillna(df['zonaloffice'])

    return df

df = load_data()

# Sidebar zonal office filter
if 'zonaloffice' in df.columns:
    st.sidebar.subheader("📍 Filter by Zonal Office")
    zonal_offices = df['zonaloffice'].dropna().unique().tolist()
    selected_zones = st.sidebar.multiselect("Select Zonal Office(s)", zonal_offices, default=zonal_offices[:3])
    df = df[df['zonaloffice'].isin(selected_zones)]

# Group by month
monthly_data = df.set_index('incidentdate').resample('ME').agg({
    'estimatedquantity': 'sum'
}).rename(columns={'estimatedquantity': 'total_volume'})

# Display data preview
st.subheader("📅 Monthly Oil Spill Volume")
st.line_chart(monthly_data)

# ADF Test
adf_stat, adf_pvalue, *_ = adfuller(monthly_data['total_volume'])
st.markdown(f"**ADF Statistic:** {adf_stat:.4f}")
st.markdown(f"**p-value:** {adf_pvalue:.4f}")

# Model and Forecast
st.subheader("🔮 Forecast Next 12 Months")
model = ARIMA(monthly_data['total_volume'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
forecast.index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME')

# Plot forecast
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_data.index, monthly_data['total_volume'], label="Historical")
ax.plot(forecast.index, forecast, label="Forecast", linestyle='--', color='red')
ax.set_title("Forecast of Monthly Oil Spill Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Total Volume (Barrels)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Developed by Onyeogali Gaxton Okobah | Powered by Streamlit, ARIMA, and Statsmodels")
