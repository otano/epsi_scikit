# epsi-scikit

Projet scolaire EPSI — pipeline d'apprentissage automatique sur les données d'accidents routiers français (2005–2010), exposé via une API Flask-RESTX.

## Prérequis

- Python 3.13
- `uv` (gestionnaire de paquets)
- Docker (optionnel, pour PostgreSQL)

## Installation

```bash
uv sync
docker compose up -d          # optionnel : démarre PostgreSQL
uv run python main.py         # entraîne les modèles
uv run python app.py          # démarre l'API sur http://localhost:9090
```

## Données

Le fichier `data/donnees_agregees_de_2005_a_2010.csv` contient ~460 000 accidents corporels de la circulation en France métropolitaine (2005–2010). Source : ONISR (Observatoire National Interministériel de la Sécurité Routière).

La variable cible `grav` est un indice de gravité calculé comme la somme des valeurs tutélaires 2010 :

| Victime        | Valeur |
| -------------- | ------ |
| Blessé léger   | 0,44   |
| Hospitalisé    | 10,8   |
| Tué            | 100    |

Exemple : un accident avec 2 blessés légers et 1 tué → `grav = 2×0.44 + 1×100 = 100.88`.

### Colonnes principales

| Colonne                       | Description                                |
| ----------------------------- | ------------------------------------------ |
| `org`                           | Organisme (1=Gendarmerie… 5=Sécurité publique) |
| `dep`                           | Code INSEE du département                  |
| `com`                           | Code INSEE de la commune                   |
| `lat`, `long`                   | Coordonnées (degrés × 100 000)              |
| `catr`                          | Catégorie de route (1=Autoroute… 9=Autre)   |
| `voie`, `v1`, `v2`              | Numéro de route et indices                 |
| `pr`, `pr1`                     | Point de référence et distance              |
| `typenumero`, `numero`          | Type et numéro d'adresse                   |
| `distancemetre`                 | Distance au numéro                         |
| `libellevoie`                   | Libellé de la voie                         |
| `coderivoli`                    | Code Rivoli (cadastre)                     |
| `ttue`, `tbg`, `tbl`, `tindm`   | Nombres de tués, hospitalisés, blessés légers, indemnes |
| `grav`                          | Indice de gravité (cible)                  |

## Modèles

Trois modèles sont entraînés par `uv run python main.py` :

| Modèle                      | Algorithme              | Features utilisées     | Performance |
| --------------------------- | ----------------------- | ---------------------- | ----------- |
| Régression (`grav`)         | RandomForestRegressor   | 11 features            | R² ≈ 0.01   |
| Classification (sévérité)   | RandomForestClassifier  | `lat`, `long`, `catr`, `distancemetre` | Acc. ≈ 72 % |
| Clustering (typologie)      | KMeans (2 clusters)     | `lat`, `long`, `catr`, `distancemetre` | 2 profils   |

> **Note** : le R² du régresseur est proche de zéro car les features disponibles (localisation, type de route) ne suffisent pas à prédire la gravité. Celle-ci dépend de facteurs absents du jeu de données (vitesse, port de ceinture, type de collision, etc.). Ce résultat est un bon exemple des limites d'un modèle entraîné sur des données partielles.

Les modèles entraînés sont sauvegardés au format `.joblib` et chargés automatiquement par l'API.

## API

### Santé

**`GET /api/health`** — Vérification que l'API répond.

```json
{"status": "ok"}
```

**`GET /api/dbhealth`** — Vérification de la connexion PostgreSQL.

```json
{"status": "ok", "postgres": "PostgreSQL 16.14 (Debian) on aarch64..."}
```

### Prédiction (régression)

**`POST /predict/`** — Prédit l'indice `grav` à partir de **11 paramètres**.

| Paramètre        | Type  | Description             |
| ---------------- | ----- | ----------------------- |
| `org`              | float | Organisme               |
| `dep`              | float | Département             |
| `com`              | float | Commune                 |
| `lat`              | float | Latitude (degrés × 100 000) |
| `long`             | float | Longitude (degrés × 100 000) |
| `catr`             | float | Catégorie de route      |
| `voie`             | float | Numéro de route         |
| `v1`               | float | Indice du numéro        |
| `pr1`              | float | Distance au PR          |
| `typenumero`       | float | Type de numéro          |
| `distancemetre`    | float | Distance au numéro      |

Requête :
```json
{
  "org": 1, "dep": 10, "com": 284,
  "lat": 4640200, "long": 491900,
  "catr": 3, "voie": 933, "v1": 0,
  "pr1": 15, "typenumero": 0, "distancemetre": 0
}
```

Réponse :
```json
{
  "gravite": 51.73,
  "features": { … }
}
```

### Prédiction (classification de sévérité)

**`POST /predict-severity/`** — Prédit la classe de sévérité à partir de **4 paramètres**.

| Paramètre        | Type  | Description                    |
| ---------------- | ----- | ------------------------------ |
| `lat`              | float | Latitude (degrés × 100 000)    |
| `long`             | float | Longitude (degrés × 100 000)   |
| `catr`             | float | Catégorie de route             |
| `distancemetre`    | float | Distance au numéro             |

Classes retournées :

| Classe   | Signification                      |
| -------- | ---------------------------------- |
| `léger`  | Victimes indemnes ou blessés légers uniquement |
| `grave`  | Au moins un blessé hospitalisé    |
| `mortel` | Au moins un tué                   |

Requête :
```json
{
  "lat": 4640200,
  "long": 491900,
  "catr": 3,
  "distancemetre": 0
}
```

Réponse :
```json
{
  "severite": "grave",
  "probabilites": {
    "léger": 0.11,
    "grave": 0.65,
    "mortel": 0.24
  },
  "features": { … }
}
```

### Typologie (clustering)

**`POST /cluster/`** — Identifie le profil d'accident à partir de **4 paramètres**.

Paramètres identiques à `/predict-severity/`.

Profils :

| Cluster | Description                                                       |
| ------- | ---------------------------------------------------------------- |
| 0       | Réseau local — routes départementales et communales, trafic modéré |
| 1       | Grand axe — autoroutes et routes nationales, trafic rapide        |

Requête :
```json
{
  "lat": 4640200,
  "long": 491900,
  "catr": 3,
  "distancemetre": 0
}
```

Réponse :
```json
{
  "cluster": 0,
  "description": "Réseau local — routes départementales et communales, trafic modéré",
  "distances": {
    "0": 0.30,
    "1": 1.17
  },
  "features": { … }
}
```

## Exemples curl

```bash
# Santé
curl http://localhost:9090/api/health

# Santé PostgreSQL
curl http://localhost:9090/api/dbhealth

# Prédiction régression
curl -X POST http://localhost:9090/predict/ \
  -H "Content-Type: application/json" \
  -d '{"org":1,"dep":10,"com":284,"lat":4640200,"long":491900,"catr":3,"voie":933,"v1":0,"pr1":15,"typenumero":0,"distancemetre":0}'

# Prédiction de sévérité (4 paramètres)
curl -X POST http://localhost:9090/predict-severity/ \
  -H "Content-Type: application/json" \
  -d '{"lat":4640200,"long":491900,"catr":3,"distancemetre":0}'

# Typologie (4 paramètres)
curl -X POST http://localhost:9090/cluster/ \
  -H "Content-Type: application/json" \
  -d '{"lat":4640200,"long":491900,"catr":3,"distancemetre":0}'
```

L'API expose aussi une interface Swagger à l'adresse `http://localhost:9090/` pour tester les endpoints interactivement.

## Structure du projet

```
.
├── main.py                 # Entraînement des modèles
├── app.py                  # Serveur API Flask-RESTX
├── docker-compose.yml      # PostgreSQL
├── pyproject.toml          # Dépendances Python
├── uv.lock                 # Lockfile des dépendances
├── .python-version         # Version Python
├── .gitignore
├── AGENTS.md               # Instructions pour l'assistant IA
├── README.md
└── data/
    ├── donnees_agregees_de_2005_a_2010.csv
    └── descriptif_des_variables_pour_le_fichier_de_2005_a_2010.pdf
```

## Licence

Projet pédagogique — données ONISR (licence ouverte Etalab).
