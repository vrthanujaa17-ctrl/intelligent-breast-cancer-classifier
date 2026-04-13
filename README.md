🧠 Intelligent Breast Cancer Classification using Machine Learning


📌 Project Overview

This project develops an intelligent decision-support system to classify breast cancer patients using the METABRIC gene expression dataset. The system predicts Estrogen Receptor (ER) status (Positive/Negative) based on clinical and gene expression features.

Multiple classical machine learning models are implemented, tuned, and compared to evaluate their effectiveness in patient-level classification.

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

🎯 Objectives

Predict ER status of breast cancer patients

Compare performance of multiple ML models

Analyze impact of parameter tuning and data splits

Generate patient-level predictions for decision support

🧠 Models Implemented

Logistic Regression

Support Vector Machine (SVM)

Decision Tree

Random Forest



## 📁 Project Structure

```bash
intelligent-breast-cancer-classifier/
│
├── data/
│   ├── raw/
│   │   └── metabric.csv
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
│   │   ├── model_results.csv
│   │   ├── parameter_experiments.csv
│   │   ├── split_experiments.csv
│   │   └── patient_predictions.csv
│   │
│   ├── plots/
│   │   ├── lr_confusion_matrix.png
│   │   ├── svm_confusion_matrix.png
│   │   ├── dt_confusion_matrix.png
│   │   ├── rf_confusion_matrix.png
│   │   ├── lr_roc_curve.png
│   │   ├── svm_roc_curve.png
│   │   ├── dt_roc_curve.png
│   │   ├── rf_roc_curve.png
│   │   ├── model_comparison_accuracy.png
│   │   ├── model_comparison_f1.png
│   │   └── model_comparison_roc_auc.png
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```



▶️ How to Run the Project:

1. Clone the repository

git clone https://github.com/your-username/intelligent-breast-cancer-classifier.git
cd intelligent-breast-cancer-classifier
2. Install dependencies

pip install -r requirements.txt

3.Run the pipeline

python src/main.py


⚙️ Pipeline Workflow

1. Data Preprocessing

Load METABRIC dataset

Handle missing values

Normalize target labels (fix dataset inconsistencies like “Positve”)

Encode categorical variables (One-hot encoding)

Preserve patient_id for final predictions

2. Feature Engineering

Variance Threshold Filtering

Principal Component Analysis (PCA)

3. Model Training & Tuning

Hyperparameter tuning using GridSearchCV

Cross-validation for robustness

4. Experiments Conducted

Parameter Experiments

Logistic Regression (C values)

SVM (kernel, C, gamma)

Decision Tree (depth, criterion)

Random Forest (estimators, depth)

Train/Test Split Experiments

70/30

80/20

90/10

5.Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

6. Visualization

Confusion Matrices

ROC Curves

Model Comparison Plots

7. Patient-Level Predictions

Final model generates predictions for individual patients

Output includes:

patient_id

Actual ER status

Predicted ER status


📊 Outputs Generated

CSV Files (in results/metrics/)

model_results.csv → Final model comparison

parameter_experiments.csv → Parameter tuning results

split_experiments.csv → Train/test split analysis

patient_predictions.csv → Patient-level classification

Plots (in results/plots/)

Confusion matrices for all models

ROC curves

Model comparison charts

📈 Key Results

All models achieved high performance (>92% accuracy)

Decision Tree showed strong generalization

SVM achieved high classification performance

Logistic Regression showed stable ROC-AUC performance

🔮 Future Work

Use deep learning models (e.g., Neural Networks)

Feature importance and explainability (SHAP, LIME)

Deployment as a web-based decision-support system

