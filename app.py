# ============================================================================ #
#                               📦 PROJECT ROOT                                #
# ============================================================================ #
# app.py
# Description  : Streamlit-based GUI launcher for the full Detoxify-Telugu NLP
#                pipeline, routing to different modules like training, annotation,
#                prediction, and summary visualizations.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# Torch Classes Patch
import sys
if 'torch.classes' not in sys.modules:
    import types
    sys.modules['torch.classes'] = types.SimpleNamespace()
sys.modules['torch.classes'].__path__ = []

# 🌐 Streamlit Detoxify App
import streamlit as st
import os
import pandas as pd

# Importing utility modules
from utils import (
    data_collection,
    data_cleaning,
    data_summary,
    model_training,
    model_evaluation,
    predict,
    text_generator,
    data_annotation,
    data_balancing,
)
from utils.YT_Scraper import render_data_collection_ui as render_youtube_ui
from utils.IG_Scraper import render_data_collection_ui as render_instagram_ui
from utils.X_Scraper import render_data_collection_ui as render_x_ui

# Define data paths
DATA_PATH = 'data/raw/toxicity_data.csv'
CLEAN_DATA_PATH = 'data/processed/toxic_data_cleaned.csv'
SCRAPED_DATA_DIR = 'data/raw/scraped'
os.makedirs(SCRAPED_DATA_DIR, exist_ok=True)

# Define separate paths for each platform's scraped data
YT_DATA_PATH = os.path.join(SCRAPED_DATA_DIR, "youtube_data.csv")
IG_DATA_PATH = os.path.join(SCRAPED_DATA_DIR, "instagram_data.csv")
X_DATA_PATH = os.path.join(SCRAPED_DATA_DIR, "x_data.csv")

# Ensuring data directory exists
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Initialize base dataset if not present
if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(columns=["Text", "Toxic_flag", "Toxic_type", "lang"])
    df_init.to_csv(DATA_PATH, index=False)

# Streamlit UI Configuration
st.set_page_config(page_title="Detoxify-Telugu Platform", layout="wide")
st.sidebar.title("🧩 Toxicity Detection Platform")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Tenglish - Telugu Generator",
        "Data Collection",
        "Data Annotation",
        "Data Cleaning",
        "Data Summary",
        "Data Balancing",
        "Model Training",
        "Model Evaluation",
        "Toxicity Prediction"
    ]
)

# Home Page
if page == "Home":
    col_logo = st.columns(3)
    with col_logo[1]:
        st.image("assets/images/detoxify_telugu_logo.png", width=600)

    st.markdown("<h1 style='text-align: center;'>Detoxify-Telugu</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Combating Toxicity in Regional Social Media Spaces</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <b>Detoxify-Telugu</b> is a pioneering initiative aimed at detecting and mitigating <b>hate speech</b>, <b>abusive language</b>, and <b>toxic behavior</b> across Telugu social media platforms. Leveraging advanced NLP techniques, this platform identifies explicit offensive content and categorizes it for better moderation and community safety.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3>🎯 Project Overview</h3>
    <div style='font-size:17px;'>
    This model serves as a foundational tool for:
    <ul>
        <li>Detecting direct abusive comments.</li>
        <li>Categorizing toxicity types.</li>
        <li>Laying the groundwork for future context-aware moderation systems.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3>👨‍💻 About the Creator</h3>
    <div style='font-size:17px;'>
    Crafted by <b>Pavan Yellathakota</b> (<a href='https://pye.pages.dev' target='_blank'>pye.pages.dev</a>) under the guidance of 
    <a href='https://www.clarkson.edu/people/boris-jukic' target='_blank'><b>Dr. Boris Jukic</b></a>, <i>Director of Business Analytics</i>, Clarkson University.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏛️ Association")
    col1, col2 = st.columns(2)
    with col1:
        st.image("assets/images/CUDS.jpg", caption="Clarkson University Data Science Dept.", width=250)
    with col2:
        st.image("assets/images/clarkson_logo.png", caption="Clarkson University", width=250)

    with st.expander("📖 **Read More: Vision & Future Roadmap**"):
        st.markdown("""
        <div style='font-size:16px;'>
        <ul>
            <li>Enhance flexibility to understand nuanced dialects.</li>
            <li>Introduce a Toxicity Scoring System for user monitoring.</li>
            <li>Expand to contextual and multilingual detection models.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    <div style='font-size:16px;'>
    🚀 <b>Detoxify-Telugu</b> is inspired by the <a href='https://pypi.org/project/detoxify/' target='_blank'><b>Detoxify Project</b></a>.
    <br><br>
    Proudly open-source and built for empowering regional content moderation.
    </div>
    """, unsafe_allow_html=True)

# Data Collection Page
elif page == "Data Collection":
    st.markdown("<h1 style='text-align: center;'>Data Collection</h1>", unsafe_allow_html=True)
    st.markdown("""
    Collect data for toxicity analysis either by manually entering text or by scraping social media platforms like YouTube, Instagram, or X (Twitter).
    """)

    # Method selection
    method = st.radio(
        "Select Data Collection Method:",
        ["Manual Entry", "Automated Scraping"]
    )

    if method == "Manual Entry":
        # Render manual data collection UI
        data_collection.render_data_collection_ui(DATA_PATH)

    elif method == "Automated Scraping":
        # Platform selection dropdown
        platform = st.selectbox(
            "Select Social Media Platform",
            ["YouTube", "Instagram", "X (Twitter)"]
        )

        # Render the appropriate scraper UI based on platform selection
        if platform == "YouTube":
            render_youtube_ui(YT_DATA_PATH)
        elif platform == "Instagram":
            render_instagram_ui(IG_DATA_PATH)
        elif platform == "X (Twitter)":
            render_x_ui(X_DATA_PATH)

# Navigation to Functional Modules
elif page == "Tenglish - Telugu Generator":
    text_generator.render_text_conversion_ui()

elif page == "Data Annotation":
    data_annotation.render_annotation_ui()

elif page == "Data Cleaning":
    data_cleaning.render_data_cleaning_ui()

elif page == "Data Summary":
    data_summary.render_data_summary_ui(CLEAN_DATA_PATH)

elif page == "Data Balancing":
    data_balancing.render_data_balancing_ui()

elif page == "Model Training":
    model_training.render_model_training_ui()

elif page == "Model Evaluation":
    model_evaluation.render_model_evaluation_ui()

elif page == "Toxicity Prediction":
    predict.render_prediction_ui()