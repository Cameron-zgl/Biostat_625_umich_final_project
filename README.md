# README: Mental Health Topic Extraction and Depression Prediction

This project consists of a two-step framework for analyzing mental health-related posts and predicting depression risk using machine learning and deep learning models. The dataset used is sourced from Zenodo and includes Reddit posts from various mental health subreddits.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Steps Overview](#steps-overview)
   - [Step 1: Topic Extraction](#step-1-topic-extraction)
   - [Step 2: Depression Prediction](#step-2-depression-prediction)
3. [Dataset Download and Setup](#dataset-download-and-setup)
4. [Installation Requirements](#installation-requirements)
5. [Workflow](#workflow)
6. [Files and Outputs](#files-and-outputs)
7. [Usage Instructions](#usage-instructions)

## Project Overview
This project aims to:
1. Extract themes/topics from mental health-related text data using clustering techniques.
2. Build a supervised learning model to predict the likelihood of depression in user posts.

## Steps Overview

### Step 1: Topic Extraction
- **Goal**: Identify key themes in mental health posts grouped by half-year intervals.
- **Techniques**:
    - Text Preprocessing (tokenization, stopword removal, lemmatization)
    - Feature Extraction using TF-IDF
    - Clustering using K-Means
    - Extracting top unigrams and bigrams for each cluster

**Input**: Preprocessed text from mental health-related posts.
**Output**: A summary of extracted topics and corresponding keywords saved to a file.

### Step 2: Depression Prediction
- **Goal**: Predict whether a given text indicates depressive tendencies.
- **Techniques**:
    - Data Annotation using:
        - Keyword-based matching
        - Sentiment analysis (VADER and Transformer-based sentiment models)
    - Feature Extraction using word embeddings (BERT)
    - Supervised Learning:
        - Logistic Regression
        - LSTM Model with BERT Embeddings
    - Evaluation metrics: Accuracy, F1-Score, Confusion Matrix

**Input**: Annotated dataset of posts labeled for depression risk.
**Output**: Trained models, evaluation metrics, and visualizations.

## Dataset Download and Setup
The dataset used in this project can be downloaded from Zenodo:

**Zenodo Link**: [Reddit Dataset](https://zenodo.org/api/records/3941387/files-archive)

### Steps to Download and Setup:
1. Run the following command in the terminal to download the dataset:
    ```bash
    wget https://zenodo.org/api/records/3941387/files-archive -O reddit_dataset.zip
    ```
2. Unzip the downloaded file into the **data/** folder:
    ```bash
    unzip reddit_dataset.zip -d data/
    ```
3. Ensure the extracted files include the required `mental_health_support.csv` and `non_mental_health.csv` files.

## Installation Requirements
Before running the project, install the required libraries:

```bash
pip install -r requirements.txt
```

**Key Libraries**:
- Python 3.8+
- PyTorch
- scikit-learn
- transformers (Hugging Face)
- NLTK
- Matplotlib

## Workflow
1. **Preprocessing and Data Preparation**:
    - Run `data_processing.ipynb` to preprocess and prepare the data.
2. **Step 1 - Topic Extraction**:
    - Run `Step1-Topic-extracition.ipynb` to extract topics and keywords using clustering.
3. **Step 2 - Depression Prediction**:
    - Run `Step2-Depression-Predict.ipynb` to train a classifier for depression risk prediction.

## Files and Outputs

### Files:
- `data/` - Folder containing the downloaded and unzipped dataset.
- `Step1-Topic-extracition.ipynb` - Notebook for topic extraction.
- `Step2-Depression-Predict.ipynb` - Notebook for depression risk prediction.
- `data_processing.ipynb` - Notebook for preprocessing the dataset.
- `lstm_depression_model.pth` - Trained LSTM model weights.
- `vocab.json` - Saved vocabulary file for inference.
- `README.md` - Project documentation (this file).

### Outputs:
- **Step 1**:
    - `merged_clusters_output.csv` - File containing topic keywords for each cluster. (There is an example in the repository.)
- **Step 2**:
    - `combined_annotated_data_updated.csv` - Annotated dataset with depression labels.
    - `classification_report.txt` - Performance metrics (precision, recall, F1-score).
    - `lstm_depression_model.pth` - Saved model for inference.

## Usage Instructions

### 1. Data Processing
Run the preprocessing notebook:
```bash
jupyter notebook data_processing.ipynb
```

### 2. Step 1: Topic Extraction
Run the topic extraction notebook:
```bash
jupyter notebook Step1-Topic-extracition.ipynb
```

**Expected Output**:
- Topic clusters and corresponding top keywords for each half-year interval.

### 3. Step 2: Depression Prediction
Run the depression prediction notebook:
```bash
jupyter notebook Step2-Depression-Predict.ipynb
```

**Expected Output**:
- Trained LSTM model with BERT embeddings.
- Performance metrics and confusion matrix.

## Model Inference
To use the trained LSTM model for inference on new text data:
1. Load the saved model:
    ```python
    import torch
    from transformers import BertTokenizer
    from model import BertLSTMClassifier

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertLSTMClassifier()
    model.load_state_dict(torch.load("lstm_depression_model.pth"))
    model.eval()
    ```
2. Predict:
    ```python
    def predict(text):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=100)
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            prediction = torch.argmax(output, dim=1)
        return 'Depression' if prediction.item() == 1 else 'Non-Depression'

    # Example
    print(predict("I feel hopeless and sad all the time."))
    ```

## Contact
If you have any issues or questions regarding the project, please feel free to contact the author or raise an issue on GitHub.

**End of README**
