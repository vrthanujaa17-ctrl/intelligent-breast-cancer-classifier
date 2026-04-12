import os
import pandas as pd
from data_preprocessing import load_data, clean_data, prepare_target, split_and_scale_data
from feature_selection import variance_filter, apply_pca
from train_models import (
    train_logistic_regression,
    train_svm,
    train_decision_tree,
    train_random_forest,
    tune_logistic_regression,
    tune_svm,
    tune_decision_tree,
    tune_random_forest
)
from evaluate import evaluate_model, cross_validate_model, save_results
from visualize import plot_confusion_matrix, plot_roc_curve, plot_model_comparison

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "metabric.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")

PARAM_RESULTS_PATH = os.path.join(RESULTS_DIR, "parameter_experiments.csv")
SPLIT_RESULTS_PATH = os.path.join(RESULTS_DIR, "split_experiments.csv")
MODEL_RESULTS_PATH = os.path.join(RESULTS_DIR, "model_results.csv")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "patient_predictions.csv")


def run_parameter_experiments(
    X_train_pca,
    X_test_pca,
    X_train_tree,
    X_test_tree,
    y_train,
    y_test
):
    results = []

    print("\nRunning parameter experiments...")

    for c_value in [0.01, 0.1, 1, 10]:
        model = train_logistic_regression(X_train_pca, y_train, C=c_value)
        metrics, _, _ = evaluate_model(model, X_test_pca, y_test, f"LR_C_{c_value}")
        metrics["Experiment_Type"] = "Parameter"
        metrics["Parameter_Setting"] = f"C={c_value}"
        results.append(metrics)

    svm_settings = [
        {"C": 1, "kernel": "linear", "gamma": "scale"},
        {"C": 1, "kernel": "rbf", "gamma": "scale"},
        {"C": 10, "kernel": "rbf", "gamma": "scale"},
    ]

    for setting in svm_settings:
        model = train_svm(
            X_train_pca,
            y_train,
            C=setting["C"],
            kernel=setting["kernel"],
            gamma=setting["gamma"]
        )
        model_name = f"SVM_{setting['kernel']}_C_{setting['C']}"
        metrics, _, _ = evaluate_model(model, X_test_pca, y_test, model_name)
        metrics["Experiment_Type"] = "Parameter"
        metrics["Parameter_Setting"] = (
            f"C={setting['C']}, kernel={setting['kernel']}, gamma={setting['gamma']}"
        )
        results.append(metrics)

    dt_settings = [
        {"max_depth": 3, "min_samples_split": 2, "criterion": "gini"},
        {"max_depth": 5, "min_samples_split": 2, "criterion": "entropy"},
        {"max_depth": None, "min_samples_split": 5, "criterion": "gini"},
    ]

    for setting in dt_settings:
        model = train_decision_tree(
            X_train_tree,
            y_train,
            max_depth=setting["max_depth"],
            min_samples_split=setting["min_samples_split"],
            criterion=setting["criterion"]
        )
        model_name = f"DT_depth_{setting['max_depth']}"
        metrics, _, _ = evaluate_model(model, X_test_tree, y_test, model_name)
        metrics["Experiment_Type"] = "Parameter"
        metrics["Parameter_Setting"] = (
            f"max_depth={setting['max_depth']}, "
            f"min_samples_split={setting['min_samples_split']}, "
            f"criterion={setting['criterion']}"
        )
        results.append(metrics)

    rf_settings = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 5},
    ]

    for setting in rf_settings:
        model = train_random_forest(
            X_train_tree,
            y_train,
            n_estimators=setting["n_estimators"],
            max_depth=setting["max_depth"],
            min_samples_split=setting["min_samples_split"]
        )
        model_name = f"RF_estimators_{setting['n_estimators']}_depth_{setting['max_depth']}"
        metrics, _, _ = evaluate_model(model, X_test_tree, y_test, model_name)
        metrics["Experiment_Type"] = "Parameter"
        metrics["Parameter_Setting"] = (
            f"n_estimators={setting['n_estimators']}, "
            f"max_depth={setting['max_depth']}, "
            f"min_samples_split={setting['min_samples_split']}"
        )
        results.append(metrics)

    return results


def run_split_experiments(X, y, patient_ids):
    results = []

    print("\nRunning train/test split experiments...")

    split_settings = [0.3, 0.2, 0.1]

    for test_size in split_settings:
        print(f"\nTesting split: train={int((1 - test_size) * 100)}/test={int(test_size * 100)}")

        (
            X_train,
            X_test,
            y_train,
            y_test,
            _,
            _,
            X_train_scaled,
            X_test_scaled,
            _
        ) = split_and_scale_data(X, y, patient_ids, test_size=test_size)

        X_train_var, X_test_var, _ = variance_filter(X_train_scaled, X_test_scaled, threshold=0.01)
        X_train_pca, X_test_pca, _ = apply_pca(X_train_var, X_test_var, n_components=0.95)

        lr_model = train_logistic_regression(X_train_pca, y_train, C=0.01)
        svm_model = train_svm(X_train_pca, y_train, C=10, kernel="rbf", gamma="scale")
        dt_model = train_decision_tree(X_train, y_train, max_depth=3, min_samples_split=2, criterion="entropy")
        rf_model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=20, min_samples_split=2)

        split_label = f"{int((1 - test_size) * 100)}/{int(test_size * 100)}"

        for model, model_name, x_test in [
            (lr_model, "Logistic Regression", X_test_pca),
            (svm_model, "SVM", X_test_pca),
            (dt_model, "Decision Tree", X_test),
            (rf_model, "Random Forest", X_test),
        ]:
            metrics, _, _ = evaluate_model(model, x_test, y_test, model_name)
            metrics["Experiment_Type"] = "Split"
            metrics["Parameter_Setting"] = f"Train/Test Split = {split_label}"
            results.append(metrics)

    return results


def run_best_model_comparison(X, y, patient_ids):
    print("\nRunning best model comparison...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        patient_id_train,
        patient_id_test,
        X_train_scaled,
        X_test_scaled,
        _
    ) = split_and_scale_data(X, y, patient_ids, test_size=0.2)

    X_train_var, X_test_var, _ = variance_filter(X_train_scaled, X_test_scaled, threshold=0.01)
    print("After variance filtering:", X_train_var.shape)

    X_train_pca, X_test_pca, _ = apply_pca(X_train_var, X_test_var, n_components=0.95)
    print("After PCA:", X_train_pca.shape)

    print("Tuning Logistic Regression...")
    best_lr, lr_params = tune_logistic_regression(X_train_pca, y_train)

    print("Tuning SVM...")
    best_svm, svm_params = tune_svm(X_train_pca, y_train)

    print("Tuning Decision Tree...")
    best_dt, dt_params = tune_decision_tree(X_train, y_train)

    print("Tuning Random Forest...")
    best_rf, rf_params = tune_random_forest(X_train, y_train)

    print("\nBest LR params:", lr_params)
    print("Best SVM params:", svm_params)
    print("Best DT params:", dt_params)
    print("Best RF params:", rf_params)

    lr_cv_mean, lr_cv_std = cross_validate_model(best_lr, X_train_pca, y_train)
    svm_cv_mean, svm_cv_std = cross_validate_model(best_svm, X_train_pca, y_train)
    dt_cv_mean, dt_cv_std = cross_validate_model(best_dt, X_train, y_train)
    rf_cv_mean, rf_cv_std = cross_validate_model(best_rf, X_train, y_train)

    print(f"\nLR CV Accuracy: {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")
    print(f"SVM CV Accuracy: {svm_cv_mean:.4f} ± {svm_cv_std:.4f}")
    print(f"DT CV Accuracy: {dt_cv_mean:.4f} ± {dt_cv_std:.4f}")
    print(f"RF CV Accuracy: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")

    final_results = []

    model_artifacts = {
        "lr": (best_lr, X_test_pca, y_test, patient_id_test),
        "svm": (best_svm, X_test_pca, y_test, patient_id_test),
        "dt": (best_dt, X_test, y_test, patient_id_test),
        "rf": (best_rf, X_test, y_test, patient_id_test),
    }

    for model, model_name, x_test, cv_mean, cv_std in [
        (best_lr, "Logistic Regression", X_test_pca, lr_cv_mean, lr_cv_std),
        (best_svm, "SVM", X_test_pca, svm_cv_mean, svm_cv_std),
        (best_dt, "Decision Tree", X_test, dt_cv_mean, dt_cv_std),
        (best_rf, "Random Forest", X_test, rf_cv_mean, rf_cv_std),
    ]:
        metrics, _, report = evaluate_model(model, x_test, y_test, model_name)
        metrics["Experiment_Type"] = "Final Comparison"
        metrics["Parameter_Setting"] = "Best tuned model"
        metrics["CV Accuracy Mean"] = cv_mean
        metrics["CV Accuracy Std"] = cv_std
        final_results.append(metrics)

        print(f"\n{model_name} Report:\n{report}")

    return final_results, model_artifacts


def save_patient_predictions(model_artifacts):
    """
    Save patient-level classification output using the best-performing model.
    Uses SVM here because it had the highest test accuracy.
    """
    best_model, X_test_best, y_test_best, patient_id_test = model_artifacts["svm"]

    y_pred = best_model.predict(X_test_best)

    predictions_df = pd.DataFrame({
        "patient_id": patient_id_test.values,
        "Actual_ER_Status": ["Positive" if val == 1 else "Negative" for val in y_test_best],
        "Predicted_ER_Status": ["Positive" if val == 1 else "Negative" for val in y_pred]
    })

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPatient predictions saved to: {PREDICTIONS_PATH}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_path = DATA_FILE_PATH
    target_col = "er_status_measured_by_ihc"

    df = load_data(file_path)
    print("Original shape:", df.shape)

    df = clean_data(df)
    print("Cleaned shape:", df.shape)

    X, y, patient_ids = prepare_target(df, target_col)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        patient_id_train,
        patient_id_test,
        X_train_scaled,
        X_test_scaled,
        _
    ) = split_and_scale_data(X, y, patient_ids, test_size=0.2)

    X_train_var, X_test_var, _ = variance_filter(X_train_scaled, X_test_scaled, threshold=0.01)
    print("After variance filtering:", X_train_var.shape)
    X_train_pca, X_test_pca, _ = apply_pca(X_train_var, X_test_var, n_components=0.95)
    print("After PCA:", X_train_pca.shape)

    parameter_results = run_parameter_experiments(
        X_train_pca, X_test_pca, X_train, X_test, y_train, y_test
    )
    save_results(parameter_results, PARAM_RESULTS_PATH)

    split_results = run_split_experiments(X, y, patient_ids)
    save_results(split_results, SPLIT_RESULTS_PATH)

    final_results, model_artifacts = run_best_model_comparison(X, y, patient_ids)
    final_df = save_results(final_results, MODEL_RESULTS_PATH)

    save_patient_predictions(model_artifacts)

    lr_model, lr_x, lr_y, _ = model_artifacts["lr"]
    svm_model, svm_x, svm_y, _ = model_artifacts["svm"]
    dt_model, dt_x, dt_y, _ = model_artifacts["dt"]
    rf_model, rf_x, rf_y, _ = model_artifacts["rf"]

    plot_confusion_matrix(
        lr_model, lr_x, lr_y,
        "Logistic Regression Confusion Matrix",
        os.path.join(PLOTS_DIR, "lr_confusion_matrix.png")
    )
    plot_confusion_matrix(
        svm_model, svm_x, svm_y,
        "SVM Confusion Matrix",
        os.path.join(PLOTS_DIR, "svm_confusion_matrix.png")
    )
    plot_confusion_matrix(
        dt_model, dt_x, dt_y,
        "Decision Tree Confusion Matrix",
        os.path.join(PLOTS_DIR, "dt_confusion_matrix.png")
    )
    plot_confusion_matrix(
        rf_model, rf_x, rf_y,
        "Random Forest Confusion Matrix",
        os.path.join(PLOTS_DIR, "rf_confusion_matrix.png")
    )

    plot_roc_curve(
        lr_model, lr_x, lr_y,
        "Logistic Regression ROC Curve",
        os.path.join(PLOTS_DIR, "lr_roc_curve.png")
    )
    plot_roc_curve(
        svm_model, svm_x, svm_y,
        "SVM ROC Curve",
        os.path.join(PLOTS_DIR, "svm_roc_curve.png")
    )
    plot_roc_curve(
        dt_model, dt_x, dt_y,
        "Decision Tree ROC Curve",
        os.path.join(PLOTS_DIR, "dt_roc_curve.png")
    )
    plot_roc_curve(
        rf_model, rf_x, rf_y,
        "Random Forest ROC Curve",
        os.path.join(PLOTS_DIR, "rf_roc_curve.png")
    )

    plot_model_comparison(
        final_df,
        metric="Accuracy",
        save_path=os.path.join(PLOTS_DIR, "model_comparison_accuracy.png")
    )
    plot_model_comparison(
        final_df,
        metric="F1-Score",
        save_path=os.path.join(PLOTS_DIR, "model_comparison_f1.png")
    )
    plot_model_comparison(
        final_df,
        metric="ROC-AUC",
        save_path=os.path.join(PLOTS_DIR, "model_comparison_roc_auc.png")
    )

    print("\nFinal Tuned Model Comparison:\n", final_df)
    print(f"\nPlots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()