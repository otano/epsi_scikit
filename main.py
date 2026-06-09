import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "data/donnees_agregees_de_2005_a_2010.csv"

TARGET = "grav"

LEAKAGE_COLS = {"ttue", "tbg", "tbl", "tindm"}
NON_NUMERIC_COLS = {"v2", "pr", "numero", "libellevoie", "coderivoli"}

FEATURE_COLS = [
    "org",
    "dep",
    "com",
    "lat",
    "long",
    "catr",
    "voie",
    "v1",
    "pr1",
    "typenumero",
    "distancemetre",
]

SEVERITY_LABELS = ["léger", "grave", "mortel"]

CLUSTER_FEATURES = ["lat", "long", "catr", "distancemetre"]


def severity_class(grav):
    if grav == 0.44:
        return 0
    elif grav < 100:
        return 1
    else:
        return 2


def load_data(file_path):
    df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")
    return df


def preprocess_data(df, target_column):
    df = df.dropna(subset=[target_column])
    X = df[FEATURE_COLS]
    y = df[target_column]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, y, imputer, scaler


def split_datasets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_regressor(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_classifier(X_train, y_train_classes):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_classes)
    return clf


def train_clustering(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"  MSE: {mse}")
    print(f"  R²: {r2}")
    return mse, r2


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(file_path)
    print(f"Shape: {df.shape}")

    print("Preprocessing...")
    X, y, imputer, scaler = preprocess_data(df, TARGET)
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    X_train, X_test, y_train, y_test = split_datasets(X, y)

    # --- Regressor ---
    print("\n--- Regressor (predict gravité) ---")
    reg = train_regressor(X_train, y_train)
    evaluate(reg, X_test, y_test)

    # --- Classifier ---
    print("\n--- Classifier (prédire sévérité) ---")
    y_classes = np.array([severity_class(v) for v in y])
    yc_train, yc_test = train_test_split(
        y_classes, test_size=0.2, random_state=42
    )
    clf = train_classifier(X_train, yc_train)
    pred_classes = clf.predict(X_test)
    acc = accuracy_score(yc_test, pred_classes)
    print(f"  Accuracy: {acc:.3f}")
    print(classification_report(yc_test, pred_classes, target_names=SEVERITY_LABELS))

    # --- Clustering ---
    print("\n--- Clustering (KMeans) on lat/long/catr/distancemetre ---")
    df_cluster = df.dropna(subset=[TARGET])
    # Remove extreme coordinate outliers that create singleton clusters
    df_cluster = df_cluster[
        (df_cluster["lat"].abs() < 1e6) & (df_cluster["long"].abs() < 1e6)
    ]
    cluster_imputer = SimpleImputer(strategy="median")
    X_cluster_raw = cluster_imputer.fit_transform(df_cluster[CLUSTER_FEATURES])
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster_raw)
    kmeans, cluster_labels = train_clustering(X_cluster_scaled)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for c, count in cluster_counts.items():
        print(f"  Cluster {c}: {count} samples")
        center = pd.DataFrame(X_cluster_scaled, columns=CLUSTER_FEATURES).groupby(cluster_labels).mean().loc[c]
        print(f"    Center (scaled): {center.round(2).to_dict()}")

    # Save everything
    joblib.dump(reg, "model.joblib")
    joblib.dump(clf, "classifier.joblib")
    joblib.dump(kmeans, "kmeans.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(imputer, "imputer.joblib")
    joblib.dump(cluster_imputer, "cluster_imputer.joblib")
    joblib.dump(cluster_scaler, "cluster_scaler.joblib")
    print("\nSaved: model.joblib, classifier.joblib, kmeans.joblib, scaler.joblib, imputer.joblib, cluster_imputer.joblib, cluster_scaler.joblib")
