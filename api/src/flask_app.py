from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)
db_connection_string = 'postgres://{username}:{password}@{host}:{port}/{database}'.format(
    username=os.getenv('POSTGRES_USERNAME'), password=os.getenv('POSTGRES_PASSWORD'), host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'), database=os.getenv('POSTGRES_DATABASE'))
app.config['SQLALCHEMY_DATABASE_URI'] = db_connection_string
db = SQLAlchemy(app)
migrate = Migrate(app, db)
