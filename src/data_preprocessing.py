import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    low_memory=False avoids mixed-type warnings in large datasets.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by:
    1. Dropping unnecessary ID columns
    2. Keeping the target column safe
    """
    df = df.copy()

    # Drop patient ID explicitly
    df = df.drop(columns=["patient_id"], errors="ignore")

    # Drop other ID-like columns, but do NOT drop the target column
    protected_cols = {"er_status_measured_by_ihc"}
    id_cols = [col for col in df.columns if "id" in col.lower() and col not in protected_cols]
    df = df.drop(columns=id_cols, errors="ignore")

    return df


def prepare_target(df: pd.DataFrame, target_col: str):
    """
    Prepare features X and target y.

    This version:
    - handles misspelled target values like 'Positve'
    - normalizes text labels
    - maps binary target values to 0/1
    - drops rows with missing/unusable target values
    - fills missing feature values
    - converts categorical features to numeric
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    print("\nRaw target value counts:")
    print(df[target_col].value_counts(dropna=False).head(20))

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col]).copy()

    # Normalize target text
    raw_target = df[target_col].astype(str).str.strip().str.lower()

    # Robust mapping, including dataset typo: 'positve'
    target_mapping = {
        "positive": 1,
        "positve": 1,
        "pos": 1,
        "1": 1,
        "yes": 1,
        "true": 1,

        "negative": 0,
        "neg": 0,
        "0": 0,
        "no": 0,
        "false": 0,
    }

    y = raw_target.map(target_mapping)

    # Keep only rows where mapping worked
    valid_mask = y.notna()
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    print("\nMapped target value counts:")
    print(y.value_counts())

    if y.nunique() < 2:
        unique_raw = sorted(raw_target.unique().tolist())[:30]
        raise ValueError(
            "Target mapping produced fewer than 2 classes. "
            f"Please inspect the target column '{target_col}'. "
            f"Sample normalized values: {unique_raw}"
        )

    # Separate features and target
    X = df.drop(columns=[target_col])

    # Fill missing values in features only
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            mode_series = X[col].mode()
            if not mode_series.empty:
                X[col] = X[col].fillna(mode_series.iloc[0])
            else:
                X[col] = X[col].fillna("unknown")

    # Convert categorical columns to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split the data into train and test sets.
    Also returns scaled versions for models like Logistic Regression and SVM.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler