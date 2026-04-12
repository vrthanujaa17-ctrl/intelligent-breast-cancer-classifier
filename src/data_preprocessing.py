import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path, low_memory=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset by dropping unnecessary ID-like columns,
    but keep patient_id and the target column.
    """
    df = df.copy()

    protected_cols = {"patient_id", "er_status_measured_by_ihc"}
    id_cols = [col for col in df.columns if "id" in col.lower() and col not in protected_cols]
    df = df.drop(columns=id_cols, errors="ignore")

    return df


def prepare_target(df: pd.DataFrame, target_col: str):
    """
    Prepare features X, target y, and patient_ids.
    Keeps the real patient_id for later prediction output,
    but excludes it from model training features.
    Handles typo in dataset label: 'Positve'.
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if "patient_id" not in df.columns:
        raise ValueError("Expected 'patient_id' column not found in dataset.")

    print("\nRaw target value counts:")
    print(df[target_col].value_counts(dropna=False).head(20))

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col]).copy()

    # Normalize target values
    raw_target = df[target_col].astype(str).str.strip().str.lower()

    target_mapping = {
        "positive": 1,
        "positve": 1,   # typo in dataset
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

    # Keep only mapped rows
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

    # Preserve patient IDs for later reporting
    patient_ids = df["patient_id"].copy()

    # Separate features and target
    X = df.drop(columns=[target_col, "patient_id"])

    # Fill missing values
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            mode_series = X[col].mode()
            if not mode_series.empty:
                X[col] = X[col].fillna(mode_series.iloc[0])
            else:
                X[col] = X[col].fillna("unknown")

    # Convert categorical columns to numeric
    X = pd.get_dummies(X, drop_first=True)

    return X, y, patient_ids


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split the data into train/test sets and scale features.
    Also keeps aligned patient IDs for train and test sets.
    """
    X_train, X_test, y_train, y_test, patient_id_train, patient_id_test = train_test_split(
        X,
        y,
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        patient_id_train,
        patient_id_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    )