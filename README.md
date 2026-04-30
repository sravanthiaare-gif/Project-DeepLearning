# Automated Consumer Complaint Classification using DistilBERT

## Project Overview
This project focuses on automating the categorization of large-scale consumer complaint narratives. By leveraging state-of-the-art Natural Language Processing (NLP) and Deep Learning techniques, the system classifies complaints into specific categories to streamline resolution workflows.

## Key Features
* **Large-Scale Data Handling:** Processed and engineered features for a **1.6 GB dataset** containing over 120,000 records.
* **Deep Learning Architecture:** Implemented a fine-tuned **DistilBERT** (Transformer) model to capture complex semantic context and intent.
* **High-Dimensional Optimization:** Managed a feature space exceeding **1,000,000 dimensions** using advanced vectorization and sparse matrix techniques.
* **Performance Benchmarking:** Outperformed traditional Machine Learning baselines (Gradient Boosting/XGBoost) in both accuracy and inference speed.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, Hugging Face Transformers (DistilBERT)
* **Machine Learning:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy, NLTK
* **Deployment/UI:** Streamlit (Optional if used)

## Methodology
1.  **Preprocessing:** Text cleaning, stop-word removal, and tokenization optimized for Transformer inputs.
2.  **Vectorization:** Implementation of TF-IDF and specialized tokenization for high-dimensional data management.
3.  **Model Training:** Fine-tuned the pre-trained DistilBERT model on the custom complaint dataset using batch processing to manage memory constraints.
4.  **Evaluation:** Validated results using Precision, Recall, and F1-score to ensure balanced classification across all categories.

## Results
The implementation of DistilBERT provided a significant improvement in capturing the nuances of consumer language, handling the 1.6 GB dataset efficiently while maintaining high prediction accuracy.
