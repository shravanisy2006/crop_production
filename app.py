import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Crop Dashboard", layout="wide")

# LOAD DATA
df = pd.read_csv("crop_production.csv")

# DATA CLEANING (Exp 9)
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

st.title("🌾 Crop Production Analytics Dashboard")

# ===================== SUMMARY (Exp 6) =====================
st.subheader("📊 Summary Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Average Production", round(df['Production'].mean(),2))
col3.metric("Max Production", df['Production'].max())

# ===================== DATA TABLE =====================
st.subheader("📋 Dataset")
st.dataframe(df.head(100))

# ===================== ADD DATA (Exp 7) =====================
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

        st.success("✅ Data Added Successfully!")

# ===================== EDA (Exp 8) =====================
st.subheader("📈 Visualizations")

if st.button("Generate Graphs"):
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Production'], ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Production'], ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax3)
    st.pyplot(fig3)

# ===================== REGRESSION (Exp 11) =====================
st.subheader("📉 Regression Analysis")

if st.button("Run Regression"):
    X = df[['Area']]
    y = df['Production']

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    st.write("R² Score:", round(r2,3))

# ===================== PCA (Exp 12) =====================
st.subheader("📊 PCA Analysis")

if st.button("Run PCA"):
    features = df[['Area', 'Production']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    result = pca.fit_transform(scaled)

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    ax.scatter(result[:,0], result[:,1])
    ax.set_title("PCA Plot")
    st.pyplot(fig)

# ===================== CLUSTERING (Exp 13) =====================
st.subheader("🔵 Clustering")

if st.button("Run K-Means"):
    X = df[['Area', 'Production']]

    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Area'], df['Production'], c=df['Cluster'])
    ax.set_title("K-Means Clustering")
    st.pyplot(fig)

    st.dataframe(df[['Area','Production','Cluster']].head())
