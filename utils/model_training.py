# ============================================================================ #
#                      📄 MODEL TRAINING MODULE (Main)                        #
# ============================================================================ #
# Filename     : model_training.py
# Description  : Handles the model training pipeline including data loading,
#                preprocessing, training loop, evaluation, and model saving.
# Author       : PAVAN YELLATHAKOTA (pye.pages.dev)
# Created Date : APR 2025
# Project      : Toxicity Detection / Classification Platform
# ============================================================================ #

import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from collections import Counter

class ToxicityDataset(Dataset):
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

def render_model_training_ui():
    st.title("🧠 Train Toxicity Classification Model")

    task_type = st.radio("Choose Model Type:", ["Binary", "Multi-Class"], horizontal=True)
    data_file = f"data/training/{'binary' if task_type == 'Binary' else 'multi'}/dataset_{'binary' if task_type == 'Binary' else 'multiclass'}.csv"
    model_dir = f"models/{task_type}"

    if not os.path.exists(data_file):
        st.error(f"❌ Dataset not found at `{data_file}`. Please balance the data first.")
        return

    df = pd.read_csv(data_file)
    if df.empty:
        st.error("❌ Loaded dataset is empty.")
        return

    st.subheader("📊 Preview Data")
    st.dataframe(df.head(5))

    if task_type == "Binary":
        df['label'] = df['Toxic_flag'].apply(lambda x: 1 if x else 0)
        num_labels = 2
        class_weights_tensor = None
    else:
        class_labels = {
            "none": 0, "threatening": 1, "sexual_abuse": 2, "profanity_generic": 3,
            "religious_caste_slur": 4, "common_insult": 5, "harassment_bullying": 6,
            "mixed_toxicity": 7, "gender_targeted": 8, "films_fan_war": 9, "political_toxicity": 10
        }
        df['Toxicity_Class'] = df.apply(lambda row: "none" if not row['Toxic_flag'] else str(row['Toxic_type']).lower(), axis=1)
        df['label'] = df['Toxicity_Class'].map(class_labels)
        num_labels = len(class_labels)

        label_counts = Counter(df['label'].dropna())
        total = sum(label_counts.values())
        class_weights = [total / (len(class_labels) * label_counts.get(i, 1)) for i in range(num_labels)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    df = df.dropna(subset=['label'])
    if len(df) < 10:
        st.error("❌ Not enough samples for training.")
        return

    st.subheader("🛠️ Model & Training Settings")

    model_options = {
        "Tiny (1M – 10M)": ["prajjwal1/bert-tiny", "google/bert_uncased_L-2_H-128_A-2"],
        "Small (10M – 25M)": ["prajjwal1/bert-mini"],
        "Medium (50M – 100M+)": ["distilbert-base-uncased", "bert-base-uncased"]
    }

    param_range = st.selectbox("Parameter Size Range:", list(model_options.keys()))
    model_choice = st.selectbox("Choose Model Variant:", model_options[param_range])

    # Try to preview actual number of parameters
    with st.spinner("Loading model to count parameters..."):
        try:
            temp_model = AutoModelForSequenceClassification.from_pretrained(model_choice, num_labels=num_labels)
            total_params = sum(p.numel() for p in temp_model.parameters())
            st.success(f"🔢 Model `{model_choice}` has approximately **{total_params:,}** parameters.")
            del temp_model
        except Exception as e:
            st.warning(f"Could not load model to estimate parameters. Reason: {str(e)}")

    with st.sidebar:
        st.header("⚙️ Training Configuration")
        epochs = st.slider("Epochs", 1, 20, 5)
        learning_rate = st.number_input("Learning Rate", value=3e-5, format="%e")
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        max_len = st.selectbox("Max Sequence Length", [64, 128, 256], index=1)
        use_gpu = st.checkbox("Use GPU if available", value=True)
        st.warning("⚠️ Choose smaller models for faster iterations. Avoid accidental long runs.")

    if st.button("🚀 Start Training"):
        st.info("⏳ Training started...")

        texts = df['Text'].tolist()
        labels = df['label'].tolist()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, train_size=0.8, stratify=labels, random_state=42
        )

        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_choice, num_labels=num_labels, torch_dtype=torch.float32
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        num_training_steps = epochs * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        training_losses = []
        progress_bar = st.progress(0)
        model.train()
        step = 0

        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                step += 1
                progress_bar.progress(min(step / num_training_steps, 1.0))

            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)
            st.write(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Save Model
        out_path = os.path.join(model_dir, model_choice.replace('/', '_'))
        os.makedirs(out_path, exist_ok=True)
        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        st.success(f"✅ Model saved to `{out_path}`")

        plot_training_loss(training_losses)

def plot_training_loss(losses):
    st.subheader("📈 Training Loss Curve")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1), losses, marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Epochs")
    plt.grid(True)
    st.pyplot(fig)
