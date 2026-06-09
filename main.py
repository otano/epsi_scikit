import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Exemple de stratégie
from sklearn.metrics import mean_squared_error, r2_score

file_path = "data/donnees_agregees_de_2005_a_2010.csv"


# Chargement des données
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Normalisation (Scaling)
# Note : La normalisation ajuste l'échelle des données.
# Pour "écarter" les données non représentatives, on utilise souvent le filtrage d'outliers.
def preprocess_data(df, target_column):
    # Suppression des valeurs manquantes
    df = df.dropna()

    X = df.drop(columns=[target_column])  # Caractéristiques
    y = df[target_column]  # Cible à prédire

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Identification apprentissage (Train) et vérification (Test)
def split_datasets(X, y):
    # test_size=0.2 signifie 20% pour la vérification, 80% pour l'apprentissage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Choix de la stratégie et Entraînement
def train_model(X_train, y_train):
    # Utilisation de Random Forest (Open-source, robuste, large communauté)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Vérification de la justesse
def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Erreur Quadratique Moyenne (MSE) : {mse}")
    print(f"Score R2 (Précision) : {r2}")

    # Visualisation simple avec Plotly
    fig = px.scatter(
        x=y_test,
        y=predictions,
        labels={"x": "Réel", "y": "Prédiction"},
        title="Réel vs Prédiction",
    )
    fig.show()


# --- Exécution ---

# df = load_data(path)
# X, y = preprocess_data(df, "nom_colonne_cible")
# X_train, X_test, y_train, y_test = split_datasets(X, y)
# model = train_model(X_train, y_train)
# evaluate(model, X_test, y_test)
