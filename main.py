import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = "data/donnees_agregees_de_2005_a_2010.csv"

TARGET = "grav"

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


def train_classifier(X_train, y_train_classes):
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train_classes)
    return clf


def train_clustering(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    return kmeans, labels


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(file_path)
    print(f"Shape: {df.shape}")

    # --- Classifier (on only 4 features: lat, long, catr, distancemetre) ---
    print("\n--- Classifier (prédire sévérité, 4 features) ---")
    df_clf = df.dropna(subset=[TARGET])
    df_clf = df_clf[(df_clf["lat"].abs() < 1e6) & (df_clf["long"].abs() < 1e6)]
    clf_imputer = SimpleImputer(strategy="median")
    X_clf_raw = clf_imputer.fit_transform(df_clf[CLUSTER_FEATURES])
    clf_scaler = StandardScaler()
    X_clf = clf_scaler.fit_transform(X_clf_raw)
    y_classes = np.array([severity_class(v) for v in df_clf[TARGET]])
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_classes, test_size=0.2, random_state=42
    )
    clf = train_classifier(Xc_train, yc_train)
    pred_classes = clf.predict(Xc_test)
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
    joblib.dump(clf, "classifier.joblib")
    joblib.dump(kmeans, "kmeans.joblib")
    joblib.dump(cluster_imputer, "cluster_imputer.joblib")
    joblib.dump(cluster_scaler, "cluster_scaler.joblib")
    joblib.dump(clf_imputer, "clf_imputer.joblib")
    joblib.dump(clf_scaler, "clf_scaler.joblib")
    print("\nSaved: classifier.joblib, kmeans.joblib, cluster_imputer.joblib, cluster_scaler.joblib, clf_imputer.joblib, clf_scaler.joblib")
