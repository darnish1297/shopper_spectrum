# -----------------------------
# shopper_spectrum_app.py
# -----------------------------

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Customer Segmentation Model
# -----------------------------
try:
    with open(r"C:\Users\Darnish S\shopper_spectrum_project\customer_segmentation_model.pkl", 'rb') as f:
        kmeans_model = pickle.load(f)
    with open(r"C:\Users\Darnish S\shopper_spectrum_project\scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Customer segmentation model or scaler not found. Please save them first.")
    st.stop()

# -----------------------------
# Load Product Dataset
# -----------------------------
try:
    product_df = pd.read_csv(r"C:\Users\Darnish S\shopper_spectrum_project\online_retail.csv")
except FileNotFoundError:
    st.warning("CSV not found. Using mock dataset for testing.")
    product_df = pd.DataFrame({
        'Description': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 'Product F']
    })

# -----------------------------
# Clean product descriptions and handle missing values
# -----------------------------
product_df['Description_clean'] = product_df['Description'].str.strip().str.lower().fillna('')

# -----------------------------
# Deduplicate products for recommendation
# -----------------------------
product_df_unique = product_df.drop_duplicates(subset=['Description_clean']).reset_index(drop=True)

# -----------------------------
# Vectorize product descriptions
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(product_df_unique['Description_clean'])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Shopper Spectrum App")

# -----------------------------
# Module 1: Product Recommendation
# -----------------------------
st.header("🎯 Product Recommendation")

product_input = st.text_input("Enter Product Name:")

if st.button("Get Recommendations"):
    # Clean input
    product_input_clean = product_input.strip().lower()

    # Check if product exists
    if product_input_clean not in product_df_unique['Description_clean'].values:
        st.warning("Product not found in dataset!")
    else:
        idx = product_df_unique[product_df_unique['Description_clean'] == product_input_clean].index[0]

        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        # Get top 5 most similar products excluding the input
        similar_indices = cosine_sim.argsort()[::-1]
        similar_indices = [i for i in similar_indices if i != idx][:5]

        recommended = product_df_unique['Description'].iloc[similar_indices].tolist()

        st.success("Recommended Products:")
        for i, prod in enumerate(recommended, start=1):
            st.write(f"{i}. {prod}")

# -----------------------------
# Module 2: Customer Segmentation
# -----------------------------
st.header("🎯 Customer Segmentation")

recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (number of purchases)", min_value=0)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, format="%.2f")

if st.button("Predict Cluster"):
    customer_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans_model.predict(customer_scaled)[0]
    cluster_labels = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
    segment = cluster_labels.get(cluster, "Unknown")
    st.success(f"The customer belongs to the '{segment}' segment.")
