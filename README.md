# Detoxify-Telugu: A Fine-Tuned BERT-Based Language Model for Hate Speech Detection in Telugu & Tenglish

### A Streamlit-Based NLP Platform for Regional Hate Speech Classification


---

## 🧭 Overview

**Detoxify-Telugu** is a thoughtful platform designed to detect toxic content across Telugu, Tenglish (Telugu-English code-mixed), and English social media inputs. It supports binary (Toxic vs. Non-Toxic) and multi-class (11 toxicity types + "none") classification using fine-tuned BERT-based models. The end-to-end pipeline, powered by a Streamlit UI, enables technical and non-technical users to scrape, annotate, train, evaluate, and predict toxicity with ease.

This platform empowers both technical and non-technical users with:

* Intuitive UI for data annotation, cleaning, model training, and evaluation.
* Real-time and batch toxicity detection via text or CSV inputs
* Keyword-based auto-annotation for efficient labeling
* Data scraping from YouTube

---

## ✨ Features

* 🚀 End-to-end NLP pipeline via Streamlit
* ⚙️  Fine-tuned BERT models for binary and multi-class toxicity detection.
* 📊 Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
* 🧪 Real-time single and batch CSV predictions for moderation.
* 🧼 Built-in data cleaning, balancing, and annotation.
* 💬 Rule-based Tenglish-to-Telugu transliteration for preprocessing.
* 📥 Selenium-based YouTube comment scraping (headless mode).

---

## 🗂️ Project Structure

```
Detoxify_Telugu/
├── assets/
├── config/
├── data/
│   ├── processed/
│   │   └── toxic_data_cleaned.csv
│   ├── raw/
│   │   ├── scraped/
│   │   │   └── toxicity_data.csv
│   ├── testing_data/
│   │   └── testing.csv
│   ├── training/
│   │   ├── binary/
│   │   │   └── dataset_binary.csv
│   │   └── multi/
│   │       └── dataset_multiclass.csv
├── docs/
├── logs/
├── models/
│   ├── Binary/
│   │   └── google_bert_uncased_L-4_H-256_A-4/
│   │   └── prajjwal1_bert-tiny/
│   └── Multi-Class/
│       ├── google_bert_uncased_L-4_H-256_A-4/
│       ├── prajjwal1_bert-mini/
│       └── prajjwal1_bert-tiny/
├── utils/
│   ├── __pycache__/
│   ├── data_annotation.py
│   ├── data_balancing.py
│   ├── data_cleaning.py
│   ├── data_collection.py
│   ├── data_summary.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── predict.py
│   ├── YT_Scraper.py
│   └── ...........................
├── venv/
├── .gitignore
├── app.py
├── DetoxifyTelugu.html
├── project.aurdino
├── README.md
└── requirements.txt

```

---

## 🛠 Installation

### 📦 Requirements

* Python 3.8+
* pip
* Google Chrome + ChromeDriver (or) Firefox + GeckoDriver

### 🔧 Setup Instructions

```bash
# Step 1: Clone the repo
https://github.com/YOUR_USERNAME/detoxify-telugu.git
cd detoxify-telugu

# Step 2: Set up virtual environment
python -m venv venv
source venv/bin/activate        # (Windows: venv\Scripts\activate)

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Launch app
streamlit run app.py
```

---

## 🚀 Functional Modules

| Module              | Description                                        |
| -------------------|----------------------------------------------------|
| Data Collection     | Scrapes YouTube comments using YT_Scraper.py       |
| Data Cleaning       | Normalizes text, handles Tenglish, removes noise   |
| Data Annotation     | Supports rule-based and manual labeling            |
| Data Balancing      | Equalizes class distribution for training          |
| Model Training      | Fine-tunes BERT models with configurable settings  |
| Model Evaluation    | Computes Accuracy, Precision, Recall, F1, matrices |
| Toxicity Prediction | Real-time or CSV-based toxicity classification     |
| Tenglish Generator  | Converts Tenglish to Telugu script for consistency |


---

## ✅ Supported Models

| Category | Model Name                      | Parameters |
|----------|----------------------------------|------------|
| Tiny     | prajjwal1/bert-tiny             | ~4.3M      |
| Small    | prajjwal1/bert-mini             | ~29M       |
| Small    | google/bert_uncased_L-4_H-256_A-4 | ~4.3M    |


Models are selected based on resource availability. Training parameters (e.g., epochs, learning rate, batch size) are tunable via the Streamlit UI. Larger models like distilbert-base-uncased (~66M parameters) are planned for future enhancements.

---

## 📈 Model Performance Report

The models were evaluated on a Telugu/Tenglish dataset (~8,000 records) for both **binary** and **multi-class** toxicity detection tasks. Below are the detailed results:

---

### 🔹 Binary Classification

- **Model Used**: `google/bert_uncased_L-4_H-256_A-4`
- **Accuracy**: 85.62%
- **Precision**:  
  - Toxic: 85.7%  
  - Non-Toxic: 85.65%
- **Recall**:  
  - Toxic: 93.4%  
  - Non-Toxic: 85.62%
- **F1-Score**:  
  - Toxic: 84.5%  
  - Overall: 85.62%

✅ **Insights**:  
This model exhibits strong and reliable performance, making it well-suited for **live moderation** of Telugu, Tenglish, and English text inputs.

🆚 `prajjwal1/bert-tiny` also performed closely with **84.72% accuracy**, and showed even **higher Non-Toxic precision** at **93.8%**, but slightly lower Toxic class performance.

---

### 🔹 Multi-Class Classification (11 Toxicity Types + "none")

- **Best Model**: `prajjwal1/bert-mini`
- **Accuracy**: 47.64%
- **Precision (macro)**: 0.571  
- **Recall (macro)**: 0.476  
- **F1-Score (macro)**: 0.491

#### 📊 Other Model Performances:
| Model                                | Accuracy | F1 Score |
|--------------------------------------|----------|----------|
| `google/bert_uncased_L-4_H-256_A-4` | 45.30%   | 0.404    |
| `prajjwal1/bert-tiny`               | 34.18%   | 0.237    |

---

### ⚠️ Challenges

- **Class Imbalance**:
  - Only **1 class** has more than **1,000 samples**
  - Another class has **700+**
  - Remaining classes range between **400–600**
  - Skews predictions toward **"none"** or dominant labels

- **Semantic Overlap**:
  - Confusion seen between:
    - `mixed_toxicity`
    - `gender_targeted`
    - `films_fan_war`
    - `political_toxicity`

- **Linguistic Diversity**:
  - Dataset includes:
    - Telugu (native script)
    - Tenglish (Telugu in Latin script)
    - Standard English
  - Adds **linguistic noise**, especially hard for smaller models like `bert-tiny`  
  - Example: `bert-tiny` scored **zero precision** for the class `threatening`

---

### 🔮 Future Plans

- Expand annotated dataset (aim for 20K+ diverse examples)
- Address class imbalance via:
  - Oversampling
  - Synthetic augmentation
- Multilingual & cross-lingual fine-tuning (e.g., `indicBERT`, `distilbert-base-uncased`)
- Add language detection and script normalization pre-processing
- Evaluate on real-world moderation scenarios

---

## 🧠 How It Works (Visual Overview)

```
[Scrape] → [Annotate] → [Clean] → [Balance] → [Train] → [Evaluate] → [Predict]
```

Each step is managed via the **Streamlit UI**, with modular scripts located in the [`utils/`](./utils) directory.

---

## 📘 Documentation

Detailed documentation is available in the [`/docs`](./docs) folder, covering:

- 📚 **Research Literature**  
  Insights into BERT models and their application to toxicity detection in multilingual contexts.

- 📊 **Evaluation Reports**  
  Includes confusion matrices, performance metrics, and key training observations.

- 🌐 **Tenglish Transliteration**  
  Notes on challenges and methods used to normalize Telugu-English mixed inputs.

---

### 📄 Visual Setup Guide

- Open [`index.html`](./index.html) for a **step-by-step walkthrough** of the system.
- It contains annotated screenshots explaining each module and how to use them.
- ▶️ A **YouTube video walkthrough** (linked inside the `/docs` folder) demonstrates the full Streamlit UI and pipeline flow.

---

## ⚠️ Dataset Disclaimer

**Important:**  
This repository only includes a **sanitized and minimized dataset** for demonstration purposes.

Due to the presence of **explicit and potentially offensive content** in the original dataset, it has **not been publicly released**.

🔒 If you require access to the full dataset for academic or research purposes, please contact the author directly to discuss terms of use.

---

## 📫 Author & Contact

**Pavan Yellathakota**  
🎓 Clarkson University  
📧 [pavanyellathakota@gmail.com](mailto:pavanyellathakota@gmail.com)  
🔗 [https://pye.pages.dev](https://pye.pages.dev)

---

## 🧾 License & Credits

- 💡 Inspired by [Detoxify](https://github.com/unitaryai/detoxify)  
- 🤗 Built with [HuggingFace Transformers](https://huggingface.co/transformers)  
- 📺 UI powered by [Streamlit](https://streamlit.io)  
- 🔍 Scraping module powered by [Selenium](https://www.selenium.dev)

---
