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


@health_ns.route("/health")
class Health(Resource):
    @health_ns.marshal_with(health_model)
    def get(self):
        return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)
