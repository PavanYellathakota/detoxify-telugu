# ============================================================================ #
#                         📉 MODEL EVALUATION MODULE                          #
# ============================================================================ #
# Filename     : model_evaluation.py
# Description  : Loads a fine-tuned model and evaluates it on validation data
#                using metrics like accuracy, F1-score, and confusion matrix.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

# 📄 model_evaluation.py (Updated)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def render_model_evaluation_ui():
    st.title("📊 Evaluate Trained Toxicity Model")

    model_type = st.radio("🧠 Select Model Type:", ["Binary", "Multi-Class"], horizontal=True)
    model_root = f"models/{model_type}"
    data_file = f"data/training/{'binary' if model_type == 'Binary' else 'multi'}/dataset_{'binary' if model_type == 'Binary' else 'multiclass'}.csv"

    available_models = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]

    if not available_models:
        st.error(f"❌ No models found under `{model_root}`")
        return

    selected_model = st.selectbox("🔍 Select Model for Evaluation:", available_models)
    if not selected_model:
        return

    if st.button("🚀 Start Evaluation"):
        model_path = os.path.join(model_root, selected_model)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        num_labels = model.config.num_labels
        task_type = "Binary" if num_labels == 2 else "Multi-Class"
        st.info(f"🔎 Loaded **{task_type}** classification model with `{num_labels}` label(s).")

        if not os.path.exists(data_file):
            st.error(f"❌ Evaluation dataset not found at `{data_file}`.")
            return

        df = pd.read_csv(data_file)
        if df.empty:
            st.error("Evaluation dataset is empty.")
            return

        if task_type == "Binary":
            df['label'] = df['Toxic_flag'].apply(lambda x: 1 if x else 0)
            target_names = ['Non-Toxic', 'Toxic']
        else:
            class_labels = {
                "none": 0, "threatening": 1, "sexual_abuse": 2, "profanity_generic": 3,
                "religious_caste_slur": 4, "common_insult": 5, "harassment_bullying": 6,
                "mixed_toxicity": 7, "gender_targeted": 8, "films_fan_war": 9, "political_toxicity": 10
            }
            df['Toxicity_Class'] = df.apply(lambda row: "none" if row['Toxic_flag'] in [False, 0] else str(row['Toxic_type']).lower(), axis=1)
            df['label'] = df['Toxicity_Class'].map(class_labels)
            df = df[df['label'].notnull()]
            target_names = list(class_labels.keys())

        df = df.dropna(subset=['label'])
        texts = df['Text'].tolist()
        labels = df['label'].astype(int).tolist()

        from sklearn.model_selection import train_test_split
        _, val_texts, _, val_labels = train_test_split(texts, labels, train_size=0.8, stratify=labels, random_state=42)

        val_dataset = ToxicDataset(val_texts, val_labels, tokenizer, max_len=128)
        val_loader = DataLoader(val_dataset, batch_size=16)

        preds, true_labels_list = [], []
        st.info("Evaluating model. Please wait...")

        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                if task_type == "Binary":
                    if logits.shape[-1] == 2:
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        preds.extend([1 if p[1] >= 0.5 else 0 for p in probs])
                    else:
                        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                        preds.extend([1 if p >= 0.5 else 0 for p in np.atleast_1d(probs)])
                else:
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds.extend(np.argmax(probs, axis=1))

                true_labels_list.extend(labels.cpu().numpy())

        acc = accuracy_score(true_labels_list, preds)
        prec = precision_score(true_labels_list, preds, average='weighted', zero_division=0)
        rec = recall_score(true_labels_list, preds, average='weighted', zero_division=0)
        f1 = f1_score(true_labels_list, preds, average='weighted', zero_division=0)

        st.success(f"🎯 Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(true_labels_list, preds, labels=list(range(len(target_names))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

        report_df = pd.DataFrame(classification_report(
            true_labels_list,
            preds,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )).transpose()
        st.dataframe(report_df.round(3))
