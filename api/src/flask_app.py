from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_swagger_ui import get_swaggerui_blueprint
from yaml import Loader, load
import os

app = Flask(__name__)
SWAGGER_URL = '/api/v1/docs'
API_URL = 'swagger/swagger.yaml'
swagger_yaml = load(open(API_URL, 'r'), Loader=Loader)
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'spec': swagger_yaml
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
db_connection_string = 'postgres://{username}:{password}@{host}:{port}/{database}'.format(
    username=os.getenv('POSTGRES_USERNAME'), password=os.getenv('POSTGRES_PASSWORD'), host=os.getenv('POSTGRES_HOST'),
    port=os.getenv('POSTGRES_PORT'), database=os.getenv('POSTGRES_DATABASE'))
app.config['SQLALCHEMY_DATABASE_URI'] = db_connection_string
db = SQLAlchemy(app)
migrate = Migrate(app, db)
