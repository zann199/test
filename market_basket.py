import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

# Konfigurasi halaman
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Load dataset
@st.cache_data  # Cache data untuk meningkatkan performa
def load_data():
    df = pd.read_csv("bread basket.csv")
    df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
    return df

df = load_data()

# Preprocessing data
df["month"] = df['date_time'].dt.month
df["day_of_week"] = df['date_time'].dt.weekday

# Mapping nilai bulan dan hari
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April", 
    5: "May", 6: "June", 7: "July", 8: "August", 
    9: "September", 10: "October", 11: "November", 12: "December"
}

day_map = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 
    4: "Friday", 5: "Saturday", 6: "Sunday"
}

df["month"] = df["month"].map(month_map)
df["day_of_week"] = df["day_of_week"].map(day_map)

# Judul aplikasi
st.title("🛒 Market Basket Analysis Menggunakan Algoritma Apriori")
st.markdown("---")

# Sidebar untuk parameter
st.sidebar.header("Parameter Analisis")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.02, 0.001)
metric = st.sidebar.selectbox("Metric", ["lift", "confidence", "support"])
min_threshold = st.sidebar.slider("Minimum Threshold", 0.5, 2.0, 1.0, 0.1)

# Tampilkan data
st.header("📊 Overview Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transaksi", df['Transaction'].nunique())
col2.metric("Total Item", df['Item'].nunique())
col3.metric("Total Records", len(df))

if st.checkbox("Tampilkan Data Mentah"):
    st.dataframe(df)

# Analisis item populer
st.header("📈 Item Paling Populer")
item_counts = df['Item'].value_counts().head(10)
st.bar_chart(item_counts)

# Analisis berdasarkan waktu
st.header("🕒 Analisis Berdasarkan Waktu")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Berdasarkan Bulan")
    month_counts = df['month'].value_counts()
    # Urutkan sesuai urutan bulan
    month_order = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    month_counts = month_counts.reindex(month_order)
    st.bar_chart(month_counts)

with col2:
    st.subheader("Berdasarkan Hari")
    day_counts = df['day_of_week'].value_counts()
    # Urutkan sesuai urutan hari
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = day_counts.reindex(day_order)
    st.bar_chart(day_counts)

# Persiapan data untuk asosiasi aturan
st.header("🔍 Analisis Asosiasi dengan Algoritma Apriori")

# Buat one-hot encoding untuk algoritma apriori
@st.cache_data
def prepare_data(df):
    transaction_list = df.groupby(['Transaction'])['Item'].apply(list).values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transaction_list).transform(transaction_list)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    return df_encoded

df_encoded = prepare_data(df)

# Progress bar untuk menunjukkan proses sedang berjalan
with st.spinner('Menghitung itemset yang sering muncul...'):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

if not frequent_itemsets.empty:
    with st.spinner('Menghasilkan aturan asosiasi...'):
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    # Format aturan agar lebih mudah dibaca
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    st.success(f"Ditemukan {len(rules)} aturan asosiasi!")
    
    # Tampilkan aturan
    st.subheader("Aturan Asosiasi")
    st.dataframe(rules.sort_values(metric, ascending=False).head(20))
    
    # Filter aturan
    st.subheader("Filter Aturan")
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05)
    filtered_rules = rules[rules['confidence'] >= min_confidence]
    st.write(f"Aturan dengan confidence ≥ {min_confidence}: {len(filtered_rules)} aturan")
    st.dataframe(filtered_rules.sort_values('confidence', ascending=False))
    
else:
    st.warning("Tidak ada itemset yang memenuhi minimum support. Coba nilai support yang lebih rendah.")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan Streamlit • Market Basket Analysis • Apriori Algorithm")