# ============================================================================ #
#                           ⛏️  DATA COLLECTION MODULE                        #
# ============================================================================ #
# Filename     : data_collection.py
# Description  : UI module to manually or programmatically collect new text data.
#                Used for dataset expansion, validation, and manual entry.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# import necessary libraries
import streamlit as st
import pandas as pd
import os

# Function to render the data collection UI
def render_data_collection_ui(csv_file):
    st.title("📝 Toxic Words/Sentences Collector")

    # Initialize CSV if not exists
    if not os.path.exists(csv_file):
        df_init = pd.DataFrame(columns=["Text", "Toxic_flag", "Toxic_type", "lang"])
        df_init.to_csv(csv_file, index=False)

    # Initialize session state for clearing text
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""

    # User selects Toxic or Non-Toxic
    toxic_flag = st.radio("Select Type:", options=["Toxic", "Non-Toxic"])

    # If Toxic, choose Toxic Type
    if toxic_flag == "Toxic":
        toxic_type = st.selectbox(
            "Select Toxic Type:",
            [
                "Threatening",
                "Sexual_Abuse",
                "Profanity_Generic",
                "Religious_Caste_Slur",
                "Common_Insult",
                "Harassment_Bullying",
                "Mixed_Toxicit"
            ]
        )
        toxic_flag_value = True
    else:
        toxic_type = "none"
        toxic_flag_value = False

    # Language selection
    lang = st.selectbox("Select Language:", ["Telugu", "Tenglish", "English"])

    # Text input using session state
    st.session_state.text_input = st.text_area("Enter Text (Comma Separated or Single Sentence):", value=st.session_state.text_input)

    # Add to CSV Button
    if st.button("Add to CSV"):
        if st.session_state.text_input.strip() == "":
            st.warning("Please enter some text before adding.")
        else:
            entries = [item.strip().strip('"').strip("'") for item in st.session_state.text_input.split(",") if item.strip() != ""]

            new_entries = []
            for entry in entries:
                new_entries.append({
                    "Text": entry,
                    "Toxic_flag": toxic_flag_value,
                    "Toxic_type": toxic_type,
                    "lang": lang
                })

            if new_entries:
                df = pd.read_csv(csv_file)
                df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
                df.to_csv(csv_file, index=False)
                st.success(f"{len(new_entries)} entries added to CSV successfully!")

                st.subheader("Recently Added Entries")
                st.dataframe(pd.DataFrame(new_entries))

                st.session_state.text_input = ""
            else:
                st.warning("No valid entries found to add.")
    
    # Display the last 20 entries from the CSV
    if st.checkbox("Show Entire CSV Data"):
        df = pd.read_csv(csv_file)
        st.dataframe(df.tail(20))
