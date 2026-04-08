import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix"):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()


def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    if not hasattr(model, "predict_proba"):
        print(f"{title}: model does not support predict_proba.")
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_model_comparison(results_df, metric="Accuracy"):
    plt.figure(figsize=(8, 6))
    plt.bar(results_df["Model"], results_df[metric])
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()