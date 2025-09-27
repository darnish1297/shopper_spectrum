# shopper_spectrum
Shopper Spectrum: A Python-based Streamlit app for customer segmentation using RFM analysis and product recommendations using TF-IDF similarity.

# Shopper Spectrum

**Shopper Spectrum** is a Python-based **retail analytics project** that enables businesses to:

* Understand customer behavior through **RFM-based customer segmentation**
* Provide **personalized product recommendations** based on product descriptions

The app is built using **Python, pandas, scikit-learn, and Streamlit**.

---

## **Features**

### 1️⃣ Product Recommendation

* User inputs a product name.
* Recommends **5 similar products** using **TF-IDF + cosine similarity**.
* Handles missing or duplicate data automatically.
* Provides results in a **styled list or card view**.

### 2️⃣ Customer Segmentation

* User inputs:

  * **Recency** (days since last purchase)
  * **Frequency** (number of purchases)
  * **Monetary** (total spend)
* Predicts **customer segment**:

  * **High-Value** – recent, frequent, and big spenders
  * **Regular** – steady purchasers but not premium
  * **Occasional** – rare or occasional purchasers
  * **At-Risk** – haven’t purchased in a long time
* Uses **KMeans clustering** on standardized RFM values.

---

## **Project Structure**

```
shopper_spectrum/
├── data/                     # Dataset
│   └── online_retail.csv
├── models/                   # Saved models and scaler
│   ├── customer_segmentation_model.pkl
│   └── scaler.pkl
├── notebooks/                # EDA and model development
│   └── jupyter.ipynb
├── streamlit_app/            # Streamlit application
│   └── streamlitt.py
├── requirements.txt          # Python dependencies
└── README.md                 # Project description
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/shopper_spectrum.git
cd shopper_spectrum/streamlit_app
```

2. Install dependencies:

```bash
pip install -r ../requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlitt.py
```

---

## **Dataset**

* `online_retail.csv` – UCI Online Retail dataset (2022–2023)
* Columns include:
  `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

---

## **Model Details**

### Customer Segmentation

* **Algorithm:** KMeans clustering
* **Features:** Recency, Frequency, Monetary (RFM)
* **Cluster Labels:** High-Value, Regular, Occasional, At-Risk

### Product Recommendation

* **Algorithm:** TF-IDF vectorization + Cosine similarity
* **Input:** Product description
* **Output:** Top 5 similar products

---


## **Notes**

* For small datasets, **silhouette score** may not be defined; the app handles this safely.
* Use **consistent RFM scaling** for accurate cluster predictions.
