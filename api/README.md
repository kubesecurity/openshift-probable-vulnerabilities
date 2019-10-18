## API for GO CVEs

### Basic structure

Code lies in `src/` directory. Openshift template lies in `openshift/` directory and swagger specification lies in `swagger/` directory. The schema diagram for the database that backs up the API is present in `diagram/` directory.

### How to run the API on your local machine

#### Prerequisites

1. Postgres
2. Python (3.x)

#### Steps

1. Install the requirements by running `pip install -r requirements.txt`.
2. We are using `flask-sqlalchemy` and `flask-migrate` tools to interact with Postgres.
3. Follow the [documentation](https://flask-migrate.readthedocs.io/en/latest/) on `flask-migrate` to initialize and run the migration command to migrate the schema. You will need to do the following sub-steps in order to migrate the DB successfully:
    1. Point the `SQLALCHEMY_DATABASE_URI` to point to the right database URI. You can do that by exporting the [environment variables](src/flask_app.py#L20-22).
    2. You need to run `export FLASK_APP='src/flask_app.py'` to point to the flask application.
    3. You will need to edit `env.py` generated when you run `flask db init` in the `migrations/` directory to include all the models that we have defined. You can do that by editing the import statement to import db and edit the target metadata:
    ```Python
    from models import db
    target_metadata = db.Model.metadata
    ```
4. Once you have done the above steps you can then proceed to migrate the DB by running `flask db migrate` and `flask db upgrade`.
5. After the migration is done, initial schema is setup. You can then run the `rest_api.py` to run the API.

### How to deploy API on devcluster

1. First deploy a postgres container by following the instructions [here](http://people.redhat.com/jrivera/openshift-docs_preview/openshift-origin/glusterfs-review/using_images/db_images/postgresql.html#using-images-db-images-postgresql).
2. Please edit the environment variables located at `openshift/template.yaml` to point to the right credentials for postgres and also change the image name if necessary.
3. Now you can look at the template file located at `openshift/template.yaml` and deploy the API on dev cluster.
4. In order to migrate the schema to your dev cluster postgres instance, please follow these steps:
    1. Login to the dev cluster using `oc login`.
    2. Then use port forwarding mechanism to port forward the postgres connection to your localhost. Your command will look something like `oc port-forward  postgresql-94-centos7-3-56rrw :5432`.
    3. After that make sure to export [environment variables](src/flask_app.py#L20-22).
    4. Then run the steps that were mentioned before.


### How to work with the API

You can render the [swagger specification](http://openshift-probable-vulnerabilities-aagshah-fabric8-analytics.devtools-dev.ext.devshift.net/api/v1/docs/) to see the available operations with the API. The url of the swagger spec is for the API that's hosted on my dev cluster instance. You can go to `/api/v1/docs` for your deployed API. 