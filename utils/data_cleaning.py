# ============================================================================ #
#                           🧹 DATA CLEANING MODULE                           #
# ============================================================================ #
# Filename     : data_cleaning.py
# Description  : Deduplicates, cleans, and validates toxic dataset before training.
#                Removes empty/null values and ensures label consistency.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# import necessary libraries
import streamlit as st
import pandas as pd
import os

# Function to render the data cleaning UI
def render_data_cleaning_ui(data_folder='data/raw'):
    st.title("🧹 Advanced Toxic Data Cleaning & Merging Tool")

    cleaned_csv_path = os.path.join('data/processed', "toxic_data_cleaned.csv")

    # List CSV files
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    if not csv_files:
        st.error(f"No CSV files found in `{data_folder}` folder.")
        return

    st.subheader("📂 Select CSV Files to Merge & Clean")
    selected_files = st.multiselect("Choose one or more CSV files:", csv_files)

    if not selected_files:
        st.info("Please select at least one CSV file to proceed.")
        return

    # Merge Selected CSVs
    dfs = []
    for file in selected_files:
        df = pd.read_csv(os.path.join(data_folder, file))
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    st.success(f"Merged {len(selected_files)} file(s). Total Rows: {merged_df.shape[0]}")

    st.write("### Preview of Merged Data")
    st.dataframe(merged_df.head(10))

    st.markdown("---")
    st.subheader("🚮 Select Columns to Remove")

    # Allow user to select columns to drop
    columns_to_remove = st.multiselect("Select unnecessary columns to drop:", merged_df.columns.tolist(),
                                       default=[col for col in merged_df.columns if col not in ['Text', 'Toxic_flag', 'Toxic_type']])

    st.markdown("---")
    st.subheader("🛠️ Select Cleaning Options")

    remove_duplicates = st.checkbox("Remove Duplicate Entries (based on Text)", value=True)
    convert_lowercase = st.checkbox("Convert English/Tenglish Text to Lowercase", value=True)
    normalize_spaces = st.checkbox("Normalize Extra Spaces", value=True)

    if st.button("🚀 Clean & Save Data"):
        cleaned_df = merged_df.copy()

        # Drop selected columns
        cleaned_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        st.success(f"Dropped columns: {columns_to_remove}")

        # Validate essential columns
        required_cols = ['Text', 'Toxic_flag', 'Toxic_type']
        missing_cols = [col for col in required_cols if col not in cleaned_df.columns]

        if missing_cols:
            st.error(f"Missing critical columns required for model training: {missing_cols}")
            return

        initial_count = cleaned_df.shape[0]

        # Perform Cleaning
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates(subset=['Text'])
        
        if convert_lowercase:
            cleaned_df['Text'] = cleaned_df.apply(
                lambda row: row['Text'].lower() if isinstance(row['Text'], str) else row['Text'],
                axis=1
            )

        if normalize_spaces:
            cleaned_df['Text'] = cleaned_df['Text'].apply(lambda x: ' '.join(str(x).split()))

        cleaned_count = cleaned_df.shape[0]
        duplicates_removed = initial_count - cleaned_count

        # Save Cleaned Data
        os.makedirs('data/processed', exist_ok=True)
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        st.success(f"✅ Cleaning Completed! {duplicates_removed} duplicate(s) removed.")
        st.success(f"Cleaned dataset saved as `{cleaned_csv_path}`")

        st.write(f"Final Rows After Cleaning: {cleaned_count}")
        st.dataframe(cleaned_df.head(10))

        st.info("You can now proceed to Model Training using this optimized dataset.")

    st.markdown("---")
    if st.checkbox("📄 Show Full Merged Dataset (Last 20 Rows)"):
        st.dataframe(merged_df.tail(20))
