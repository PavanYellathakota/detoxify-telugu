# ================================
# 📄 Updated model_evaluation.py
# ================================

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
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
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
    if not st.button("🚀 Start Evaluation"):
        return

    model_path = os.path.join(model_root, selected_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    task_type = "Binary" if num_labels == 2 else "Multi-Class"
    st.info(f"🔎 Loaded **{task_type}** model with `{num_labels}` classes.")

    if not os.path.exists(data_file):
        st.error("Dataset missing. Run data balancing first.")
        return

    df = pd.read_csv(data_file)
    if task_type == "Binary":
        df['label'] = df['Toxic_flag'].apply(lambda x: 1 if x else 0)
        target_names = ['Non-Toxic', 'Toxic']
    else:
        class_labels = {
            "none": 0, "threatening": 1, "sexual_abuse": 2, "profanity_generic": 3,
            "religious_caste_slur": 4, "common_insult": 5, "harassment_bullying": 6,
            "mixed_toxicity": 7, "gender_targeted": 8, "films_fan_war": 9, "political_toxicity": 10
        }
        df['Toxicity_Class'] = df.apply(lambda row: "none" if not row['Toxic_flag'] else str(row['Toxic_type']).lower(), axis=1)
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

    preds, true_labels_list, logits_all = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits_all.extend(logits.cpu().numpy())
            true_labels_list.extend(labels.cpu().numpy())
            if task_type == "Binary":
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds.extend(np.argmax(probs, axis=1))
            else:
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    st.subheader("📈 Basic Metrics")
    acc = accuracy_score(true_labels_list, preds)
    prec = precision_score(true_labels_list, preds, average='weighted', zero_division=0)
    rec = recall_score(true_labels_list, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels_list, preds, average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels_list, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(true_labels_list, preds)

    st.write(f"**Accuracy**: `{acc:.4f}`")
    st.write(f"**Precision (weighted)**: `{prec:.4f}`")
    st.write(f"**Recall (weighted)**: `{rec:.4f}`")
    st.write(f"**F1-score (weighted)**: `{f1:.4f}`")
    st.write(f"**F1-score (macro)**: `{f1_macro:.4f}`")
    st.write(f"**Matthews Correlation Coefficient**: `{mcc:.4f}`")

    # AUROC (multi-class)
    try:
        y_true_bin = np.eye(num_labels)[np.array(true_labels_list)]
        y_score_bin = F.softmax(torch.tensor(logits_all), dim=1).numpy()
        auroc = roc_auc_score(y_true_bin, y_score_bin, average='macro', multi_class='ovr')
        st.write(f"**AUROC (macro)**: `{auroc:.4f}`")
    except:
        st.warning("AUROC computation failed (class imbalance or single-class batch).")

    with st.expander("📌 Class-wise Report"):
        report_df = pd.DataFrame(classification_report(
            true_labels_list,
            preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )).transpose()
        st.dataframe(report_df.round(3))
        csv = report_df.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Report as CSV", csv, "toxicity_evaluation_report.csv", "text/csv")

    with st.expander("📉 Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(10, 6))
        cm = confusion_matrix(true_labels_list, preds, labels=list(range(len(target_names))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    with st.expander("🚫 False Positives & Negatives Per Class"):
        cm = confusion_matrix(true_labels_list, preds, labels=list(range(len(target_names))))
        fp_fn_data = []
        for i, label in enumerate(target_names):
            fn = cm[i].sum() - cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            tp = cm[i, i]
            fp_fn_data.append({"Class": label, "TP": tp, "FP": fp, "FN": fn})
        st.dataframe(pd.DataFrame(fp_fn_data))
