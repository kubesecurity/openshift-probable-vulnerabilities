## API for GO CVEs

### Basic structure

Code lies in `src/` directory. Openshift template lies in `openshift/` directory and swagger specification lies in `swagger/` directory.

### How to run the API on your local machine

#### Prerequisites

1. Postgres
2. Python (3.x)

#### Steps

1. Install the requirements by running `pip install -r requirements.txt`.
2. We are using `flask-sqlalchemy` and `flask-migrate` tools to interact with Postgres.
3. Follow the [documentation](https://flask-migrate.readthedocs.io/en/latest/) on `flask-migrate` to initialize and run the migration command to migrate the schema. You will need to do the following sub-steps in order to migrate the DB successfully:
    1. Point the `SQLALCHEMY_DATABASE_URI` to point to the right database URI. 
    2. You need to run `export FLASK_APP='src/flask_app.py'` to point to the flask application.
    3. You will need to edit `env.py` generated when you run `flask db init` in the `migrations/` directory to include all the models that we have defined. You can do that by editing the import statement to import db and edit the target metadata:
    ```Python
    from flask_app import db
    target_metadata = db.Model.metadata
    ```
4. Once you have done the above steps you can then proceed to migrate the DB.
5. After the migration is done, initial schema is setup. You can then run the `rest_api.py` to run the API.

### How to deploy API on devcluster

1. First deploy a postgres container by following the instructions [here](http://people.redhat.com/jrivera/openshift-docs_preview/openshift-origin/glusterfs-review/using_images/db_images/postgresql.html#using-images-db-images-postgresql).
2. After the deployment of you can look at the template file located at `openshift/template.yaml` and deploy the API on dev cluster.

Note: You need to run `/api/v1/insert` only once to insert the data in the freshly created db instance. If you try to delete the data using `/api/v1/delete`, then again try to insert the data, it won't work. So in case you face any issues while inserting the data make sure to drop the DB and start afresh.


### How to check your inserted data and query the DB

You can run the [API](http://openshift-probable-vulnerabilities-aagshah-fabric8-analytics.devtools-dev.ext.devshift.net/api/v1/getcve?start-date=2019-10-07&end-date=2019-10-07) to check the data that is inserted in the DB as of now. You can also just replace the call with your own API setup on your dev cluster. For more details please check the swagger spec, which you can render at [editor.swagger.io](editor.swagger.io) for a visual treat.