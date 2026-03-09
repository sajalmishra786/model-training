import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="TEP Prediction Dashboard", layout="wide")

st.title("TEP Fault Prediction Dashboard")

# Connect to database
conn = sqlite3.connect("predictions.db")

# Load data into pandas
df = pd.read_sql_query("SELECT * FROM predictions", conn)

conn.close()

# Show total records
st.subheader("Total Predictions")
st.write(len(df))

# Show data table
st.subheader("Stored Data")
st.dataframe(df)

# --- Prediction Distribution ---
st.subheader("Prediction Distribution")

prediction_counts = df["prediction"].value_counts()
# --- Total Safe and Danger Counts ---
st.subheader("Safe vs Danger Predictions")

safe_count = prediction_counts.get("safe", 0)
danger_count = prediction_counts.get("danger", 0)

col1, col2, col3 = st.columns(3)

col1.metric("Safe Predictions", safe_count)
col2.metric("Danger Predictions", danger_count)
col3.metric("Total Predictions", safe_count - danger_count)

fig1, ax1 = plt.subplots()
prediction_counts.plot(kind="bar", ax=ax1)
ax1.set_xlabel("Prediction")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# --- Trend Over Time ---
st.subheader("Prediction Trend Over Time")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

fig2, ax2 = plt.subplots()
ax2.plot(df["timestamp"], range(len(df)))
ax2.set_xlabel("Time")
ax2.set_ylabel("Cumulative Predictions")

st.pyplot(fig2)
