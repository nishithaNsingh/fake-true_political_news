# 📰 Fake News Detection using Logistic Regression

This project aims to detect whether a news article is **fake** or **real** using a machine learning model, specifically **Logistic Regression**. It leverages natural language processing (NLP) techniques to process and analyze news text data.

---

## 📌 Features

- Binary classification: Detects if a given news article is *Fake* or *Real*
- Preprocessing of text data using NLP techniques
- Model built using **Logistic Regression**
- Evaluates performance using accuracy, confusion matrix, and classification report
- Supports batch prediction or single input testing

---

## 🔧 Tech Stack

- Python 🐍  
- Scikit-learn 🤖  
- Pandas 🐼  
- NumPy 🔢  
- NLP techniques (TF-IDF Vectorizer)  
- Jupyter Notebook / Kaggle Notebook

---

## 📂 Dataset

**Dataset used:** [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

- Contains two CSV files: `Fake.csv` and `True.csv`
- Each row contains a news article, its title, and other metadata

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/nishithaNsingh/Fake-News-Detection-with-LR.git
cd fake-news-detection-lr
```

### 2. Install dependencies

```bash
pip install 

```

## ▶️ Run the Notebook
Open the Jupyter Notebook or upload the notebook to Kaggle and run all cells to train the model and test predictions.

## 🧠 Model Workflow
1. Load the dataset

2. Combine fake and real news into a single dataframe

3. Preprocess the text (remove punctuation, lowercase, stopwords, etc.)

4. Convert text to numerical vectors using TF-IDF

5. Train a Logistic Regression model

6. Evaluate model accuracy and test it on new data

## 💡 Future Improvements
1. Use more advanced models like Random Forest, XGBoost, or LSTM

2. Create a web app with Flask or Streamlit for user interaction

3. Use word embeddings like Word2Vec or BERT for better semantic understanding
