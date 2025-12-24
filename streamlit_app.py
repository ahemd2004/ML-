import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Credit Card Customer Analysis", layout="wide")

st.title("💳 Credit Card Customer Analysis")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Credit Card Customer Data.csv")

df = load_data()

# -----------------------------
# Show Data
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.write(df.describe())

# -----------------------------
# Preprocessing
# -----------------------------
features = [
    "Avg_Credit_Limit",
    "Total_Credit_Cards",
    "Total_visits_bank",
    "Total_visits_online",
    "Total_calls_made"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# KMeans Clustering
# -----------------------------
st.subheader("KMeans Clustering")

k = st.slider("Select number of clusters (k)", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels

# -----------------------------
# Silhouette Score
# -----------------------------
score = silhouette_score(X_scaled, labels)
st.write(f"**Silhouette Score:** {score:.3f}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
sns.scatterplot(
    x=df["Avg_Credit_Limit"],
    y=df["Total_Credit_Cards"],
    hue=df["Cluster"],
    palette="Set2",
    ax=ax
)
ax.set_xlabel("Average Credit Limit")
ax.set_ylabel("Total Credit Cards")

st.pyplot(fig)

# -----------------------------
# Cluster Summary
# -----------------------------
st.subheader("Cluster Summary")
st.dataframe(df.groupby("Cluster")[features].mean())
