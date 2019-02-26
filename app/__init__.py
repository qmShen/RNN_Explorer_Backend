# -*- coding: UTF-8 -*-
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
mongo = PyMongo(app)

# flask_cors: Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible.
CORS(app)

from app.routes import index
