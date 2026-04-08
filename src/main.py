from data_preprocessing import load_data, clean_data, prepare_target, split_and_scale_data
from feature_selection import variance_filter, apply_pca
from train_models import (
    tune_logistic_regression,
    tune_svm,
    tune_decision_tree,
    tune_random_forest
)
from evaluate import evaluate_model, cross_validate_model, save_results
from visualize import plot_confusion_matrix, plot_roc_curve, plot_model_comparison


def main():
    # -----------------------------
    # STEP 1: Load and clean data
    # -----------------------------
    file_path = "../data/raw/metabric.csv"
    target_col = "er_status_measured_by_ihc"

    df = load_data(file_path)
    print("Original shape:", df.shape)

    df = clean_data(df)
    print("Cleaned shape:", df.shape)

    # -----------------------------
    # STEP 2: Prepare X and y
    # -----------------------------
    X, y = prepare_target(df, target_col)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # -----------------------------
    # STEP 3: Split and scale
    # -----------------------------
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale_data(X, y)

    # -----------------------------
    # STEP 4: Feature reduction
    # -----------------------------
    X_train_var, X_test_var, var_selector = variance_filter(X_train_scaled, X_test_scaled, threshold=0.01)
    print("After variance filtering:", X_train_var.shape)

    X_train_pca, X_test_pca, pca_model = apply_pca(X_train_var, X_test_var, n_components=0.95)
    print("After PCA:", X_train_pca.shape)

    # -----------------------------
    # STEP 5: Train tuned models
    # Use PCA data for LR and SVM
    # Use original data for DT and RF
    # -----------------------------
    print("\nTuning Logistic Regression...")
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

    # -----------------------------
    # STEP 6: Cross-validation
    # -----------------------------
    lr_cv_mean, lr_cv_std = cross_validate_model(best_lr, X_train_pca, y_train)
    svm_cv_mean, svm_cv_std = cross_validate_model(best_svm, X_train_pca, y_train)
    dt_cv_mean, dt_cv_std = cross_validate_model(best_dt, X_train, y_train)
    rf_cv_mean, rf_cv_std = cross_validate_model(best_rf, X_train, y_train)

    print(f"\nLR CV Accuracy: {lr_cv_mean:.4f} ± {lr_cv_std:.4f}")
    print(f"SVM CV Accuracy: {svm_cv_mean:.4f} ± {svm_cv_std:.4f}")
    print(f"DT CV Accuracy: {dt_cv_mean:.4f} ± {dt_cv_std:.4f}")
    print(f"RF CV Accuracy: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")

    # -----------------------------
    # STEP 7: Final evaluation
    # -----------------------------
    results_list = []

    lr_results, lr_cm, lr_report = evaluate_model(best_lr, X_test_pca, y_test, "Logistic Regression")
    svm_results, svm_cm, svm_report = evaluate_model(best_svm, X_test_pca, y_test, "SVM")
    dt_results, dt_cm, dt_report = evaluate_model(best_dt, X_test, y_test, "Decision Tree")
    rf_results, rf_cm, rf_report = evaluate_model(best_rf, X_test, y_test, "Random Forest")

    results_list.extend([lr_results, svm_results, dt_results, rf_results])

    # -----------------------------
    # STEP 8: Save results
    # -----------------------------
    results_df = save_results(results_list, "../results/metrics/model_results.csv")
    print("\nFinal Results:\n", results_df)

    # -----------------------------
    # STEP 9: Visualize
    # -----------------------------
    plot_confusion_matrix(best_lr, X_test_pca, y_test, "LR Confusion Matrix")
    plot_confusion_matrix(best_svm, X_test_pca, y_test, "SVM Confusion Matrix")
    plot_confusion_matrix(best_dt, X_test, y_test, "DT Confusion Matrix")
    plot_confusion_matrix(best_rf, X_test, y_test, "RF Confusion Matrix")

    plot_roc_curve(best_lr, X_test_pca, y_test, "LR ROC Curve")
    plot_roc_curve(best_svm, X_test_pca, y_test, "SVM ROC Curve")
    plot_roc_curve(best_dt, X_test, y_test, "DT ROC Curve")
    plot_roc_curve(best_rf, X_test, y_test, "RF ROC Curve")

    plot_model_comparison(results_df, metric="Accuracy")
    plot_model_comparison(results_df, metric="F1-Score")
    plot_model_comparison(results_df, metric="ROC-AUC")

    # -----------------------------
    # STEP 10: Print classification reports
    # -----------------------------
    print("\nLogistic Regression Report:\n", lr_report)
    print("\nSVM Report:\n", svm_report)
    print("\nDecision Tree Report:\n", dt_report)
    print("\nRandom Forest Report:\n", rf_report)


if __name__ == "__main__":
    main()