import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Crop Dashboard", layout="wide")

st.title("🌾 Crop Production Analytics Dashboard")

# -------------------- LOAD DATA SAFELY --------------------
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "crop_production.csv")
    df = pd.read_csv(file_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert numeric-like columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Handle missing values safely
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            if df[col].isnull().all():
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df

df = load_data()

# -------------------- SUMMARY (Exp 6) --------------------
st.subheader("📊 Summary Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))

if "Production" in df.columns:
    col2.metric("Average Production", round(df["Production"].mean(), 2))
    col3.metric("Max Production", df["Production"].max())
else:
    col2.metric("Average Production", "N/A")
    col3.metric("Max Production", "N/A")

# -------------------- DATA TABLE --------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(100))

# -------------------- ADD DATA (Exp 7) --------------------
st.subheader("➕ Add New Record")

with st.form("add_form"):
    state = st.text_input("State")
    district = st.text_input("District")
    year = st.number_input("Year", step=1)
    season = st.text_input("Season")
    crop = st.text_input("Crop")
    area = st.number_input("Area")
    production = st.number_input("Production")

    submit = st.form_submit_button("Add Data")

    if submit:
        try:
            new_row = pd.DataFrame([{
                "State_Name": state,
                "District_Name": district,
                "Crop_Year": year,
                "Season": season,
                "Crop": crop,
                "Area": area,
                "Production": production
            }])

            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("crop_production.csv", index=False)

            st.success("✅ Data Added Successfully! Please refresh.")
        except Exception as e:
            st.error(f"Error adding data: {e}")

# -------------------- EDA (Exp 8) --------------------
st.subheader("📈 Visualizations")

if st.button("Generate Graphs"):
    try:
        numeric_df = df.select_dtypes(include=np.number)

        if "Production" in numeric_df.columns:
            fig1, ax1 = plt.subplots()
            sns.histplot(numeric_df["Production"], ax=ax1)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=numeric_df["Production"], ax=ax2)
            st.pyplot(fig2)

        if len(numeric_df.columns) > 1:
            fig3, ax3 = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, ax=ax3)
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Visualization Error: {e}")

# -------------------- REGRESSION (Exp 11) --------------------
st.subheader("📉 Regression Analysis")

if st.button("Run Regression"):
    try:
        if "Area" in df.columns and "Production" in df.columns:
            X = df[["Area"]]
            y = df["Production"]

            model = LinearRegression()
            model.fit(X, y)

            r2 = model.score(X, y)
            st.success(f"R² Score: {round(r2, 3)}")
        else:
            st.warning("Required columns not found.")
    except Exception as e:
        st.error(f"Regression Error: {e}")

# -------------------- PCA (Exp 12) --------------------
st.subheader("📊 PCA Analysis")

if st.button("Run PCA"):
    try:
        if "Area" in df.columns and "Production" in df.columns:
            features = df[["Area", "Production"]]

            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            pca = PCA(n_components=2)
            result = pca.fit_transform(scaled)

            st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

            fig, ax = plt.subplots()
            ax.scatter(result[:, 0], result[:, 1])
            ax.set_title("PCA Plot")
            st.pyplot(fig)
        else:
            st.warning("Required columns not found.")
    except Exception as e:
        st.error(f"PCA Error: {e}")

# -------------------- CLUSTERING (Exp 13) --------------------
st.subheader("🔵 Clustering (K-Means)")

if st.button("Run Clustering"):
    try:
        if "Area" in df.columns and "Production" in df.columns:
            X = df[["Area", "Production"]]

            kmeans = KMeans(n_clusters=3, n_init=10)
            df["Cluster"] = kmeans.fit_predict(X)

            fig, ax = plt.subplots()
            scatter = ax.scatter(df["Area"], df["Production"], c=df["Cluster"])
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)

            st.dataframe(df[["Area", "Production", "Cluster"]].head())
        else:
            st.warning("Required columns not found.")
    except Exception as e:
        st.error(f"Clustering Error: {e}")
