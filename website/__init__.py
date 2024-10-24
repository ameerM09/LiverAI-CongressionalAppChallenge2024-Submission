# Python package file of "website" folder
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path

from flask_login import LoginManager

def create_web_app():
    web_app = Flask(__name__)
    web_app.secret_key = ""

    from .routes import routes

    web_app.register_blueprint(routes, url_prefix = "/")

    return web_app