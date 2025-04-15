from flask import Flask
from flask_cors import CORS

from app import routes


def create_app():
    app = Flask(__name__)
    app.register_blueprint(routes.base)
    CORS(app)
    return app
