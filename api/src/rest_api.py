from sqlalchemy import and_, Date, cast, exc
from flask_app import app
from models import *
import flask


def get_model_from_type(model_type, dict_object):
    if model_type == 'repository':
        return Repositories(**dict_object)
    elif model_type == 'issue':
        return Issues(**dict_object)
    elif model_type == 'pullrequest':
        return PullRequests(**dict_object)
    elif model_type == 'commit':
        return Commits(**dict_object)
    elif model_type == 'probablecve':
        return ProbableCVE(**dict_object)


@app.route('/api/v1/insert', methods=['POST'])
def insert_data():
    body = flask.request.get_json()
    model_type = body.pop('type')
    if not model_type:
        return flask.jsonify(message="Please specify the type of record you want to insert"), 400
    try:
        model = get_model_from_type(model_type, body)
        db.session.add(model)
        db.session.commit()
    except (exc.SQLAlchemyError, AssertionError, ValueError) as e:
        return flask.jsonify(message=str(e)), 500
    return flask.jsonify(response="Data inserted"), 200


@app.route('/api/v1/delete', methods=['POST'])
def delete_data():
    body = flask.request.get_json()
    model_type = body.pop('type')
    if not model_type:
        return flask.jsonify(message="Please specify the type of record you want to delete"), 400
    model = get_model_from_type(model_type, body).__class__
    query = model.query.filter_by(**body).first_or_404(description="No {} object with the provided details found"
                                                       .format(model_type))
    db.session.delete(query)
    try:
        db.session.commit()
    except exc.SQLAlchemyError as e:
        return flask.jsonify(message=str(e)), 500
    return flask.jsonify(message='record deleted'), 200


@app.route('/api/v1/get')
def get():
    model_type = flask.request.args.get('type')
    model_id = flask.request.args.get('id')
    model = get_model_from_type(model_type, {"id": model_id}).__class__
    try:
        query = model.query.get(model_id)
    except exc.SQLAlchemyError as e:
        return flask.jsonify(message=str(e)), 500
    response = query.__dict__
    del response['_sa_instance_state']
    return flask.jsonify(response), 200


@app.route('/api/v1/getid', methods=['POST'])
def get_id():
    body = flask.request.get_json()
    model_type = body.pop('type')
    if not model_type:
        return flask.jsonify(message="Please specify the type of record you want to fetch"), 400
    model = get_model_from_type(model_type, body).__class__
    query = model.query.filter_by(**body).first_or_404(description=
                                                       "No record of type {} with the submitted details found"
                                                       .format(model_type))
    return flask.jsonify(id=query.id), 200


@app.route('/api/v1/getcve')
def get_cve():
    start_date = flask.request.args.get('start-date')
    end_date = flask.request.args.get('end-date')
    response = ProbableCVE.query.filter(and_(cast(ProbableCVE.identified_date, Date) <= end_date,
                                             cast(ProbableCVE.identified_date, Date) >= start_date)).first_or_404(
        description='CVE with the given criteria not found'
    )
    response = response.__dict__
    del response['_sa_instance_state']
    return flask.jsonify(response), 200


@app.route('/api/v1/update', methods=['POST'])
def update():
    body = flask.request.get_json()
    model_type = body.pop('type', None)
    if not model_type:
        return flask.jsonify(message="Please specify the type of record you want to update"), 400
    model_id = body.pop('id', None)
    if not model_id:
        return flask.jsonify(message="Please specify the id of the record you want to update"), 400
    model = get_model_from_type(model_type, body).__class__
    _ = model.query.filter_by(id=model_id).update(body)
    try:
        db.session.commit()
    except (exc.SQLAlchemyError, AssertionError, ValueError) as e:
        return flask.jsonify(message=str(e)), 500
    return flask.jsonify(message='CVEs updated'), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
