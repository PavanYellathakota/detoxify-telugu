# ============================================================================ #
#                           📊 DATA SUMMARY MODULE                            #
# ============================================================================ #
# Filename     : data_summary.py
# Description  : Provides class distribution, count statistics, and word frequencies
#                for exploratory data analysis (EDA) before training.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def render_data_summary_ui(csv_file):
    st.title("📊 Data Summary Dashboard")

    # Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"No dataset found at {csv_file}. Please collect some data first.")
        return

    if df.empty:
        st.warning("The dataset is empty. Please add data through the Data Collection module.")
        return

    st.markdown(f"### Total Records: {df.shape[0]}")

    st.markdown("---")

    # 1️⃣ Toxic vs Non-Toxic Distribution
    st.subheader("Toxic vs Non-Toxic Distribution")
    fig1, ax1 = plt.subplots()
    toxic_counts = df['Toxic_flag'].value_counts()
    ax1.pie(toxic_counts, labels=['Non-Toxic', 'Toxic'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("---")

    # 2️⃣ Toxic Type Distribution
    st.subheader("Toxic Type Distribution (Only Toxic Entries)")
    toxic_df = df[df['Toxic_flag'] == True]
    if not toxic_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10,5))
        sns.countplot(data=toxic_df, x='Toxic_type', order=toxic_df['Toxic_type'].value_counts().index, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.info("No toxic entries to display.")

    st.markdown("---")

    # 3️⃣ Language Distribution
    st.subheader("Language Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='lang', order=df['lang'].value_counts().index, ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")

    # 4️⃣ WordCloud for Toxic Texts
    st.subheader("WordCloud of Toxic Texts")
    if not toxic_df.empty:
        toxic_text = " ".join(toxic_df['Text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_text)
        fig4, ax4 = plt.subplots(figsize=(10,5))
        ax4.imshow(wordcloud, interpolation='bilinear')
        ax4.axis('off')
        st.pyplot(fig4)
    else:
        st.info("No toxic data available for WordCloud.")

    st.markdown("---")

    # 5️⃣ Basic Text Statistics
    st.subheader("Basic Text Statistics")
    df['text_length'] = df['Text'].astype(str).apply(len)
    st.write(f"**Average Text Length:** {df['text_length'].mean():.2f} characters")
    st.write(f"**Longest Text:** {df.loc[df['text_length'].idxmax()]['Text']}")
    st.write(f"**Shortest Text:** {df.loc[df['text_length'].idxmin()]['Text']}")

    st.markdown("---")

    # Show raw data option
    if st.checkbox("Show Raw Dataset"):
        st.dataframe(df.drop(columns=['text_length']))
