# ============================================================================ #
#                          ⚖️  DATA BALANCING MODULE                          #
# ============================================================================ #
# Filename     : data_balancing.py
# Description  : Interface for balancing class distribution in binary or
#                multi-class datasets using oversampling or undersampling.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# Import necessary libraries
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import uuid

'''  
-- > Cached Data Loader and Balancer function loads a CSV dataset and applies class balancing (binary ormulti-class) 
        using either oversampling or undersampling techniques.
-- > It is decorated with @st.cache_data to cache the output and avoid redundant computation when the same parameters are used, 
        significantly improving performance for large datasets in Streamlit apps.
'''

@st.cache_data
def load_and_balance_data(file_path, method, target_count, balance_level, strict=True, remove_cats=None):
    df = pd.read_csv(file_path)
    # Checking if the dataset is too large — saving your PC from a meltdown, bro 😅
    if len(df) > 100000:
        st.warning("⚠️ Large dataset detected. Sampling 100,000 rows to improve performance.")
        df = df.sample(n=100000, random_state=42)
    
    # Checking if the dataset has the required columns
    if 'Toxic_flag' not in df.columns:
        st.error("❌ The dataset must contain a `Toxic_flag` column.")
        return None
    
    # Cleaning Toxic_flag column
    if df['Toxic_flag'].isna().any():
        st.warning("⚠️ Missing values in `Toxic_flag`. Dropping rows with missing values.")
        df = df.dropna(subset=['Toxic_flag'])
    df['Toxic_flag'] = df['Toxic_flag'].astype(bool)
    
    # Checking if the dataset satisfies the requirements for multi-class balancing
    if balance_level == "Multi-Class (Toxic_type)":
        if 'Toxic_type' not in df.columns:
            st.error("❌ The dataset must contain a `Toxic_type` column for multi-class balancing.")
            return None
        if df['Toxic_type'].isna().any():
            st.warning("⚠️ Missing values in `Toxic_type`. Dropping rows with missing values.")
            df = df.dropna(subset=['Toxic_type'])
        
        if remove_cats:
            df = df[~df['Toxic_type'].isin(remove_cats)]
            if df.empty:
                st.error("❌ All data filtered out after removing categories. Please select fewer categories.")
                return None
    
    # Check for required columns and ensure they are present in the DataFrame
    if balance_level.startswith("Binary"):
        balanced_df = balance_binary(df, method, target_count)
        final_cols = ['Text', 'Toxic_flag']
    else:
        balanced_df = balance_multiclass(df, method, target_count, strict)
        final_cols = ['Text', 'Toxic_flag', 'Toxic_type']
    
    if balanced_df is None or balanced_df.empty:
        st.error("❌ Balancing failed: No data available after processing.")
        return None
    
    return balanced_df[final_cols]

# --------- Streamlit UI for Data Balancing ---------
def render_data_balancing_ui():
    st.title("⚖️ Data Balancing Interface")

    data_folder = 'data/processed'
    binary_output = 'data/training/binary'
    multi_output = 'data/training/multi'

    # Ensure output directories exist
    os.makedirs(binary_output, exist_ok=True)
    os.makedirs(multi_output, exist_ok=True)

    # List available CSV files in the data folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    if not csv_files:
        st.error("❌ No CSV files found in 'data/processed'. Please ensure the folder exists and contains CSV files, or upload a CSV below.")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            csv_files = ["uploaded_file.csv"]
            selected_file = csv_files[0]
            file_path = None
        else:
            return
    else:
        selected_file = st.selectbox("📂 Select CSV to Balance:", csv_files)
        file_path = os.path.join(data_folder, selected_file)
        df = pd.read_csv(file_path)

    st.subheader("📊 Current Class Distribution")
    balance_level = st.radio("⚙️ Balance At:", ["Binary Level (Toxic vs Non-Toxic)", "Multi-Class (Toxic_type)"])

    if balance_level == "Multi-Class (Toxic_type)":
        if 'Toxic_type' not in df.columns:
            st.error("❌ The dataset must contain a `Toxic_type` column for multi-class balancing.")
            return
        valid_types = df['Toxic_type'].dropna().unique().tolist()
        remove_cats = st.multiselect("🧹 Remove unwanted categories:", valid_types)
    else:
        remove_cats = None

    plot_class_distribution(df, level=balance_level)

    st.markdown("---")
    st.subheader("🛠️ Balancing Options")
    method = st.selectbox("Method:", ["Undersampling", "Oversampling"])
    target_count = st.number_input("Target Samples per Class:", min_value=100, max_value=10000, value=1000, step=100)
    strict = st.checkbox("Strict Mode (drop low-sample classes)?", value=True)

    # Default save paths
    default_save_path = os.path.join(binary_output, 'dataset_binary.csv') if balance_level.startswith("Binary") else os.path.join(multi_output, 'dataset_multiclass.csv')
    save_path = st.text_input("💾 Output file path:", value=default_save_path)

    if st.button("🚀 Preview Balanced Data"):
        balanced_df = load_and_balance_data(file_path or uploaded_file, method, target_count, balance_level, strict, remove_cats)
        if balanced_df is not None:
            st.session_state['balanced_df'] = balanced_df
            st.session_state['save_path'] = save_path
            st.session_state['balance_level'] = balance_level

    if 'balanced_df' in st.session_state:
        st.subheader("✅ Balanced Distribution Preview")
        plot_class_distribution(st.session_state['balanced_df'], level=st.session_state['balance_level'])
        st.dataframe(st.session_state['balanced_df'].head(10))

        if st.button("💾 Save Balanced Dataset"):
            try:
                os.makedirs(os.path.dirname(st.session_state['save_path']), exist_ok=True)
                st.session_state['balanced_df'].to_csv(st.session_state['save_path'], index=False)
                st.success(f"✅ Saved to: `{st.session_state['save_path']}`")
            except Exception as e:
                st.error(f"❌ Failed to save file: {str(e)}")

# Function to plot distribution of data adjusted for binary and multi-class datasets
def plot_class_distribution(df, level):
    if df.empty:
        st.error("❌ Cannot plot: Dataset is empty.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if level.startswith("Binary"):
        counts = df['Toxic_flag'].value_counts().rename({True: 'Toxic', False: 'Non-Toxic'})
        counts.plot(kind='bar', ax=ax)
        ax.set_title("Toxic vs Non-Toxic")
    else:
        counts = df['Toxic_type'].value_counts()
        counts.plot(kind='bar', ax=ax)
        ax.set_title("Toxicity Class Distribution")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# dataset balancing for binary models
def balance_binary(df, method, target_count):
    toxic = df[df['Toxic_flag'] == True]
    non_toxic = df[df['Toxic_flag'] == False]
    
    if len(toxic) == 0 or len(non_toxic) == 0:
        st.error("❌ Cannot balance: One or both classes have no samples.")
        return pd.DataFrame(columns=df.columns)
    
    if method == "Undersampling":
        toxic_bal = resample(toxic, replace=False, n_samples=min(len(toxic), target_count), random_state=42)
        non_toxic_bal = resample(non_toxic, replace=False, n_samples=min(len(non_toxic), target_count), random_state=42)
    else:
        toxic_bal = resample(toxic, replace=True, n_samples=target_count, random_state=42)
        non_toxic_bal = resample(non_toxic, replace=True, n_samples=target_count, random_state=42)
    
    return pd.concat([toxic_bal, non_toxic_bal]).sample(frac=1).reset_index(drop=True)

# dataset balancing for multi-class models
def balance_multiclass(df, method, target_count, strict=True):
    classes = df['Toxic_type'].dropna().unique()
    balanced = []
    dropped_classes = []
    
    for cls in classes:
        subset = df[df['Toxic_type'] == cls]
        if len(subset) < target_count * 0.5 and strict:
            dropped_classes.append(cls)
            continue
        if method == "Undersampling":
            resampled = resample(subset, replace=False, n_samples=min(len(subset), target_count), random_state=42)
        else:
            resampled = resample(subset, replace=True, n_samples=target_count, random_state=42)
        balanced.append(resampled)
    
    if dropped_classes:
        st.warning(f"⚠️ Dropped classes with insufficient samples: {', '.join(dropped_classes)}")
    
    non_toxic = df[df['Toxic_flag'] == False]
    if len(non_toxic) == 0:
        st.warning("⚠️ No non-toxic samples available.")
    else:
        if method == "Undersampling":
            non_toxic_bal = resample(non_toxic, replace=False, n_samples=min(len(non_toxic), target_count), random_state=42)
        else:
            non_toxic_bal = resample(non_toxic, replace=True, n_samples=target_count, random_state=42)
        balanced.append(non_toxic_bal)
    
    if not balanced:
        st.error("❌ No classes remain after balancing.")
        return pd.DataFrame(columns=df.columns)
    
    return pd.concat(balanced).sample(frac=1).reset_index(drop=True)

