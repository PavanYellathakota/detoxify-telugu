# ============================================================================ #
#                           📝 DATA ANNOTATION MODULE                         #
# ============================================================================ #
# Filename     : data_annotation.py
# Description  : Auto/manual annotation of toxic records using reference JSONs.
#                Adds `toxic_flag`, `toxic_type`, and `lang` columns to dataset.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import json
import re

# Load Toxicity Words JSON
with open('utils/toxicity_words.json', 'r', encoding='utf-8') as f:
    TOXICITY_DICT = json.load(f)

def detect_language(text):
    return "Telugu" if re.search(r'[\u0C00-\u0C7F]', text) else "Tenglish"

def keyword_classify(text):
    for tox_type, scripts in TOXICITY_DICT.items():
        for script, words in scripts.items():
            for word in words:
                if word.lower() in text.lower():
                    return tox_type
    return None

def render_annotation_ui(data_folder='data/raw'):
    st.title("✍️ Smart Data Annotation Tool")

    # Select CSV File
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found in the data folder!")
        return

    selected_file = st.selectbox("📂 Select CSV to Annotate:", csv_files)
    file_path = os.path.join(data_folder, selected_file)
    df = pd.read_csv(file_path)

    # Ensure annotation columns exist
    for col in ['toxic_flag', 'toxic_type', 'lang']:
        if col not in df.columns:
            df[col] = None

    # Choose Mode
    mode = st.radio("Choose Annotation Mode:", ["Automation ⚡", "Manual 📝"])

    if mode == "Automation ⚡":
        st.header("⚡ Automated Annotation")

        label_exists = st.checkbox("Does the dataset contain `label` or `Label` column?", value=('label' in df.columns or 'Label' in df.columns))

        if label_exists:
            label_col = 'label' if 'label' in df.columns else 'Label'
            df['toxic_flag'] = df[label_col].apply(lambda x: True if str(x).strip().lower() in ['1', 'true'] else False)
        else:
            df['toxic_flag'] = df['toxic_flag'].fillna(False).infer_objects(copy=False)

        # Auto Language Detection
        df['lang'] = df.apply(lambda row: detect_language(str(row['Text'])) if pd.isna(row['lang']) else row['lang'], axis=1)

        # Auto Toxic Type Classification
        for idx, row in df.iterrows():
            if row['toxic_flag'] and (pd.isna(row['toxic_type']) or row['toxic_type'] == ''):
                tox_type = keyword_classify(str(row['Text']))
                df.at[idx, 'toxic_type'] = tox_type if tox_type else 'Mixed_Toxicity'

        st.success("✅ Automated classification completed!")

        # Pagination for Review
        paginate_and_review(df)

    else:
        st.header("📝 Manual Annotation Mode")
        search_keyword = st.text_input("🔍 Search by keyword (optional):")
        filtered_df = df

        if search_keyword:
            filtered_df = df[df['Text'].str.contains(search_keyword, case=False, na=False)]
            st.info(f"Found {len(filtered_df)} rows containing '{search_keyword}'")

        paginate_and_manual_label(filtered_df, df)

    # Save Button
    if st.button("💾 Save Annotated Data"):
        save_name = selected_file.replace('.csv', '_annotated.csv')
        df.to_csv(os.path.join(data_folder, save_name), index=False)
        st.success(f"Annotated data saved as `{save_name}`")
        st.balloons()

# --- Pagination for Review ---
def paginate_and_review(df):
    st.subheader("🔎 Review Annotated Data")
    rows_per_page = st.selectbox("Rows per page:", [25, 50, 100], index=0)
    total_pages = (len(df) // rows_per_page) + 1

    if 'page' not in st.session_state:
        st.session_state.page = 1

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous"):
        if st.session_state.page > 1:
            st.session_state.page -= 1
    if col2.button("Next ➡️"):
        if st.session_state.page < total_pages:
            st.session_state.page += 1

    start_idx = (st.session_state.page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    review_df = df.iloc[start_idx:end_idx]
    st.dataframe(review_df[['Text', 'toxic_flag', 'toxic_type', 'lang']])

# --- Manual Annotation with Pagination ---
def paginate_and_manual_label(filtered_df, main_df):
    rows_per_page = st.selectbox("Rows per page:", [25, 50, 100], index=0)
    total_pages = (len(filtered_df) // rows_per_page) + 1

    if 'manual_page' not in st.session_state:
        st.session_state.manual_page = 1

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous", key='manual_prev'):
        if st.session_state.manual_page > 1:
            st.session_state.manual_page -= 1
    if col2.button("Next ➡️", key='manual_next'):
        if st.session_state.manual_page < total_pages:
            st.session_state.manual_page += 1

    start_idx = (st.session_state.manual_page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    for idx in filtered_df.index[start_idx:end_idx]:
        st.write(f"**Comment:** {main_df.at[idx, 'Text']}")

        toxic_flag = st.selectbox(f"Toxic? (Row {idx})", ["Toxic", "Non-Toxic"], key=f"tox_{idx}")
        if toxic_flag == "Toxic":
            tox_type = st.selectbox(f"Select Toxic Type (Row {idx})", list(TOXICITY_DICT.keys()), key=f"type_{idx}")
            main_df.at[idx, 'toxic_flag'] = True
            main_df.at[idx, 'toxic_type'] = tox_type
        else:
            main_df.at[idx, 'toxic_flag'] = False
            main_df.at[idx, 'toxic_type'] = "none"

        main_df.at[idx, 'lang'] = detect_language(str(main_df.at[idx, 'Text']))
        st.markdown("---")
