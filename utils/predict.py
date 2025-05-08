# ============================================================================ #
#                          🔮 TOXICITY PREDICT MODULE                         #
# ============================================================================ #
# Filename     : predict.py
# Description  : Streamlit-based interface for single-sentence or batch CSV
#                prediction using trained binary/multi-class toxicity models.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# import necessary libraries
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os

# ---- Render Prediction UI ----
def render_prediction_ui():
    st.title("🔮 Real-Time & Batch Toxicity Prediction")

    # --- Step 1: Select Model Type ---
    model_type = st.radio("🧠 Select Model Type:", ["Binary", "Multi-Class"], horizontal=True)
    model_root = f"models/{model_type}"

    available_models = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
    if not available_models:
        st.error(f"❌ No trained models found in `{model_root}/`. Please train a model first.")
        return

    selected_model = st.selectbox("🧩 Choose a model for prediction:", available_models)
    if not selected_model:
        return

    model_path = os.path.join(model_root, selected_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32
        )
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load model/tokenizer:\n\n{str(e)}")
        return

    num_labels = model.config.num_labels
    task_type = "Binary" if num_labels == 2 else "Multi-Class"
    st.success(f"✅ Loaded **{task_type}** model with `{num_labels}` label(s)")

    class_map = {
        0: "Non-Toxic", 1: "Threatening", 2: "Sexual_Abuse", 3: "Profanity_Generic",
        4: "Religious_Caste_Slur", 5: "Common_Insult", 6: "Harassment_Bullying", 7: "Mixed_Toxicity"
    }

    mode = st.radio("Choose Prediction Mode:", ["🔹 Single Text", "📂 Batch CSV"], horizontal=True)
    st.markdown("---")

    # --- Mode 1: Single Prediction ---
    if mode == "🔹 Single Text":
        user_input = st.text_area("📝 Enter text to predict:")
        if st.button("🚀 Predict"):
            if not user_input.strip():
                st.warning("⚠️ Please enter some text.")
                return
            run_prediction([user_input], tokenizer, model, task_type, class_map, device, show_table=False)

    # --- Mode 2: Batch Prediction ---
    else:
        uploaded_file = st.file_uploader("📂 Upload CSV with a `Text` column", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "Text" not in df.columns:
                    st.error("❌ CSV must have a column named `Text`.")
                    return
                if st.button("🚀 Predict on Uploaded CSV"):
                    predictions = run_prediction(df["Text"].tolist(), tokenizer, model, task_type, class_map, device, show_table=True)
                    df['Prediction'] = predictions
                    st.success("✅ Prediction completed.")
                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV with Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    st.markdown("---")
    st.info("This tool supports both real-time and batch toxicity classification.")


# ---- Core Prediction Logic ----
def run_prediction(texts, tokenizer, model, task_type, class_map, device, show_table=False):
    results = []
    model.eval()

    for text in texts:
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        if task_type == "Binary":
            if logits.shape[-1] == 2:
                probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
                toxic_prob = probs[1]
            else:
                toxic_prob = torch.sigmoid(logits).cpu().item()
            label = "TOXIC" if toxic_prob >= 0.5 else "NON-TOXIC"
            results.append(label)

            if show_table is False:
                confidence = toxic_prob * 100 if label == "TOXIC" else (1 - toxic_prob) * 100
                msg = f"⚠️ **TOXIC** | Confidence: {confidence:.2f}%" if label == "TOXIC" else f"✅ **NON-TOXIC** | Confidence: {confidence:.2f}%"
                st.error(msg) if label == "TOXIC" else st.success(msg)

        else:
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
            pred_class = probs.argmax()
            label = class_map.get(pred_class, "Unknown")
            results.append(label)

            if show_table is False:
                confidence = probs[pred_class] * 100
                msg = f"⚠️ **{label}** | Confidence: {confidence:.2f}%" if label != "Non-Toxic" else f"✅ **{label}** | Confidence: {confidence:.2f}%"
                st.error(msg) if label != "Non-Toxic" else st.success(msg)

    return results
