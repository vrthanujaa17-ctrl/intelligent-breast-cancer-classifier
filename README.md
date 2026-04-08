🧠 Intelligent Breast Cancer Classification using Machine Learning
📌 Project Overview

This project develops an intelligent machine learning system to classify Estrogen Receptor (ER) status in breast cancer patients using the METABRIC dataset.

The system builds a full ML pipeline including preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

👥 Team Members
Thanujaa Vudayagiri Ravindra Babu – Logistic Regression, SVM, Evaluation
Tejaswi Kasipally – EDA, Feature Selection, Decision Tree
Shreyanka Saggidi – Random Forest, Visualization, Model Comparison

📊 Dataset
Dataset: METABRIC Breast Cancer Dataset
Source: https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
Description:
~1900 patient samples
~700+ raw features (expanded to ~8000 after encoding)
Includes clinical + gene mutation features
Target: ER Status (binary classification)
⚠️ Data Issue Handled

The dataset contains a typo:

"Positve" instead of "Positive"

This is handled during preprocessing.

📁 Project Structure
intelligent-breast-cancer-classifier/
│
├── data/
│   ├── raw/
│   │   └── metabric.csv
│   └── processed/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_selection.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── main.py
│
├── results/
│   ├── metrics/
│   │   └── model_results.csv
│   ├── plots/
│   └── models/
│
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation
1. Clone the repository
git clone https://github.com/<your-username>/intelligent-breast-cancer-classifier.git
cd intelligent-breast-cancer-classifier
2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate
3. Install dependencies
pip install pandas numpy matplotlib scikit-learn seaborn

▶️ How to Run the Project
Step 1: Download dataset

Download the dataset from Kaggle and place it here:

data/raw/metabric.csv
Step 2: Run the pipeline
cd src
python main.py


🔄 Workflow
1.Data loading & cleaning
2.Target preprocessing (handling typo + mapping labels)
3.Missing value handling
4.One-hot encoding for categorical features
5.Train-test split
6.Feature reduction:
  Variance filtering
  PCA
7.Model training:
  Logistic Regression
  Support Vector Machine (SVM)
  Decision Tree
  Random Forest
8.Hyperparameter tuning (GridSearchCV)
9.Cross-validation
10.Model evaluation
11.Visualization and comparison

📈 Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.9387 |    0.9404 | 0.9827 |   0.9611 |  0.9563 |
| SVM                 |   0.9440 |    0.9527 | 0.9758 |   0.9641 |  0.9546 |
| Decision Tree       |   0.9413 |    0.9717 | 0.9516 |   0.9615 |  0.9430 |
| Random Forest       |   0.9253 |    0.9251 | 0.9827 |   0.9530 |  0.9503 |

🏆 Key Findings
SVM achieved the highest accuracy (94.4%)
Logistic Regression achieved the highest ROC-AUC (95.6%)
Decision Tree showed strong cross-validation performance
All models performed consistently well on high-dimensional data

⚠️ Challenges & Solutions
1. Data inconsistency
Issue: "Positve" typo in labels
Solution: Custom target mapping
2. High dimensionality (~8000 features)
Solution:
Variance filtering
PCA
3. Class imbalance
More positive cases than negative
Future work: apply balancing techniques

