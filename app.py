import os
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import joblib
import numpy as np
import pandas as pd
import psycopg2
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(
    app,
    version="0.1.0",
    title="epsi-scikit API",
    description="API for French road accident data ML pipeline",
)

health_ns = api.namespace("api", description="Operations")

health_model = api.model(
    "Health",
    {"status": fields.String(description="API status", example="ok")},
)

db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "epsi_scikit"),
    "user": os.getenv("DB_USER", "epsi"),
    "password": os.getenv("DB_PASS", "epsi"),
}


def get_db_connection():
    return psycopg2.connect(**db_config)


regressor = joblib.load("model.joblib")
classifier = joblib.load("classifier.joblib")
kmeans = joblib.load("kmeans.joblib")
cluster_imputer = joblib.load("cluster_imputer.joblib")
cluster_scaler = joblib.load("cluster_scaler.joblib")
scaler = joblib.load("scaler.joblib")
imputer = joblib.load("imputer.joblib")

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

CLUSTER_DESCRIPTIONS = {
    0: "Réseau secondaire — routes départementales/communales (catr ≥ 3)",
    1: "Grand axe — autoroutes et routes nationales (catr ≤ 2)",
}


def features_from_payload(payload):
    return pd.DataFrame(
        [[payload[col] for col in FEATURE_COLS]], columns=FEATURE_COLS
    )


def prepare(features_df):
    return scaler.transform(imputer.transform(features_df))


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/json")
def hello_json():
    return {"message": "<p>Hello, World!</p>"}


@health_ns.route("/health")
class Health(Resource):
    @health_ns.marshal_with(health_model)
    def get(self):
        return {"status": "ok"}


@health_ns.route("/dbhealth")
class DbHealth(Resource):
    def get(self):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0]
            cur.close()
            conn.close()
            return {"status": "ok", "postgres": pg_version}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# --- Input models ---
input_model = api.model(
    "Features",
    {col: fields.Float(description=col, required=True) for col in FEATURE_COLS},
)

cluster_input_model = api.model(
    "ClusterFeatures",
    {col: fields.Float(description=col, required=True) for col in CLUSTER_FEATURES},
)

# --- Regressor ---
reg_ns = api.namespace("predict", description="Régression de la gravité")

reg_output = api.model(
    "RegressorOutput",
    {
        "gravite": fields.Float(description="Indice de gravité prédit", example=11.67),
        "features": fields.Raw(description="Caractéristiques fournies"),
    },
)


@reg_ns.route("/")
class Predict(Resource):
    @reg_ns.expect(input_model)
    @reg_ns.marshal_with(reg_output)
    def post(self):
        data = api.payload
        X = features_from_payload(data)
        X_prepared = prepare(X)
        pred = regressor.predict(X_prepared)[0]
        return {"gravite": float(pred), "features": data}


# --- Classifier ---
clf_ns = api.namespace("predict-severity", description="Classification de la sévérité")

clf_output = api.model(
    "ClassifierOutput",
    {
        "severite": fields.String(
            description="Classe de sévérité", example="léger"
        ),
        "probabilites": fields.Raw(
            description="Probabilités par classe",
            example={"léger": 0.7, "grave": 0.2, "mortel": 0.1},
        ),
        "features": fields.Raw(description="Caractéristiques fournies"),
    },
)


@clf_ns.route("/")
class Classify(Resource):
    @clf_ns.expect(input_model)
    @clf_ns.marshal_with(clf_output)
    def post(self):
        data = api.payload
        X = features_from_payload(data)
        X_prepared = prepare(X)
        probs = classifier.predict_proba(X_prepared)[0]
        label = classifier.predict(X_prepared)[0]
        return {
            "severite": SEVERITY_LABELS[label],
            "probabilites": {
                SEVERITY_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "features": data,
        }


# --- Clustering ---
clust_ns = api.namespace("cluster", description="Typologie d'accident (clustering)")

clust_output = api.model(
    "ClusterOutput",
    {
        "cluster": fields.Integer(description="Numéro du cluster", example=0),
        "description": fields.String(
            description="Profil du cluster",
            example="Urbain / départementale (...)",
        ),
        "distances": fields.Raw(
            description="Distance aux centres de chaque cluster"
        ),
        "features": fields.Raw(description="Caractéristiques fournies"),
    },
)


@clust_ns.route("/")
class Cluster(Resource):
    @clust_ns.expect(cluster_input_model)
    @clust_ns.marshal_with(clust_output)
    def post(self):
        data = api.payload
        X = pd.DataFrame(
            [[data[col] for col in CLUSTER_FEATURES]], columns=CLUSTER_FEATURES
        )
        X_prepared = cluster_scaler.transform(cluster_imputer.transform(X))
        distances = kmeans.transform(X_prepared)[0]
        label = kmeans.predict(X_prepared)[0]
        return {
            "cluster": int(label),
            "description": CLUSTER_DESCRIPTIONS.get(
                int(label), "Profil non défini"
            ),
            "distances": {
                str(i): round(float(d), 4) for i, d in enumerate(distances)
            },
            "features": data,
        }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)
