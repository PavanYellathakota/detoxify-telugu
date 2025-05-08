# Detoxify-Telugu: A Fine-Tuned BERT-Based Language Model for Hate Speech Detection in Telugu & Tenglish

### A Streamlit-Based NLP Platform for Regional Hate Speech Classification


---

## 🧭 Overview

**Detoxify-Telugu** is a thoughtful platform designed to detect toxic content across Telugu, Tenglish, and English social media inputs. The system supports both **binary** (toxic vs. non-toxic) and **multi-class** classification modes, enabling users to train, evaluate, and predict toxicity in an end-to-end pipeline powered by **BERT-based transformer models**.

This platform empowers both technical and non-technical users with:

* Intuitive UI for training/evaluation/prediction
* Real-time toxicity detection via text/CSV
* Keyword-based auto annotation
* Data scraping from YouTube

---

## ✨ Features

* 🚀 End-to-end NLP pipeline via Streamlit
* ⚙️ BERT fine-tuning (Binary + Multi-class)
* 📊 Accuracy, F1-score, Confusion Matrix support
* 🧪 Real-time single and batch CSV predictions
* 🧼 Cleaning, balancing, annotation built-in
* 💬 Transliteration for Tenglish > Telugu
* 📥 YouTube Scraping with Selenium (Headless)

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
| ------------------- | -------------------------------------------------- |
| Data Collection     | Scrape YouTube Comments (Selenium-based)           |
| Data Cleaning       | Remove duplicates, fix spelling, detect script     |
| Data Annotation     | Rule-based / manual labeling by category           |
| Data Balancing      | Sample equal toxic and non-toxic classes           |
| Model Training      | Train custom BERT models (tiny, small, base)       |
| Model Evaluation    | Accuracy, Precision, Recall, Confusion Matrix      |
| Toxicity Prediction | Real-time or CSV-based classification              |
| Tenglish Generator  | Convert transliterated Tenglish into Telugu script |

---

## ✅ Supported Models

| Category        | Model Name              | Size   |
| --------------- | ----------------------- | ------ |
| Tiny (1M-10M)   | prajjwal1/bert-tiny     | \~4M   |
| Small (10M-50M) | prajjwal1/bert-mini     | \~11M  |
| Medium (50M+)   | distilbert-base-uncased | \~66M  |
| Base (100M+)    | bert-base-uncased       | \~110M |

Select the size & accuracy tradeoff that fits your resource availability. Training configurations such as epochs, LR, batch size are tunable via the UI.

---

## 📈 Model Performance Report

We evaluated our models on both binary and multi-class toxicity classification tasks. Here’s how they performed:

### 🔹 Binary Classification  
- **Accuracy**: 85.6%  
- **F1 Score**: 0.8562  
- **Model Used**: `google/bert_uncased_L-4_H-256_A-4`  
✅ This model performs reliably and demonstrates strong potential for binary toxicity detection across Telugu, Tenglish, and English inputs.

---

### 🔹 Multi-Class Classification  
- **Accuracy**: 34.1%  
- **F1 Score**: 0.2754  
- **Model Used**: `prajjwal1/bert-tiny`  

⚠️ We're actively working to improve the multi-class classification model. Here are a few reasons that might explain its current underperformance:

- 📉 **Limited Training Data**: The model was trained on approximately **8,000 records**, which isn’t sufficient for fine-grained multi-class detection.
- ⚖️ **Imbalanced Class Distribution**:
  - The dataset contains **11 distinct classes**.
  - Only **1 class** has more than **1,000 samples**, One more class has **700+ samples**,
  - The remaining classes each range between **400–600 records**.
- 🌐 **Linguistic Diversity**:
  - Our dataset includes a mixture of **pure Telugu script**, **Tenglish** (Telugu-English transliteration), and **standard English**.
  - This variation increases linguistic noise and can lead to misclassifications, especially when the model hasn’t seen enough diverse examples.

We plan to address these limitations in future iterations by expanding the dataset, balancing class representation, and exploring multilingual fine-tuning strategies.


---

## 🧠 How It Works (Visual Overview)

```
[Scrape] → [Annotate] → [Clean] → [Balance] → [Train] → [Evaluate] → [Predict]
```

Each step is fully modular inside the Streamlit UI.

---

---

## 📘 Documentation

More detailed documentation available in the [`/docs`](./docs) folder.

Includes:

* Research Literature  
* Evaluation Reports  
* Confusion Matrices  
* Training Observations  

📄 **Visual Setup Guide**:  
Checkout [`DetoxifyTelugu.html`](./DetoxifyTelugu.html) for a **step-by-step guide** to set up and use the Detoxify-Telugu platform via Streamlit.  
📸 It includes annotated screenshots for each functionality.  
▶️ A **YouTube walkthrough** is also attached to help you follow along visually and interact with the system effectively.


---

---

## ⚠️ Dataset Disclaimer
Important:
The dataset included in this repository is a sanitized and minimized version for demonstration purposes only.
Due to the presence of explicit and potentially offensive content in the original dataset, it is not included publicly in this repository.

🔒 To access the full dataset, please contact the author directly for request and usage terms.

---

---

## 📫 Author & Contact

**Pavan Yellathakota**
🎓 Clarkson University
📧 [pavanyellathakota@gmail.com](mailto:pavanyellathakota@gmail.com)
🔗 [https://pye.pages.dev](https://pye.pages.dev)

---

## 🧾 License & Credits

* Inspired by [Detoxify](https://github.com/unitaryai/detoxify)
* Uses [HuggingFace Transformers](https://huggingface.co/transformers/)
* Interface built on [Streamlit](https://streamlit.io)
