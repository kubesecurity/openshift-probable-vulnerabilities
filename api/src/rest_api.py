from sqlalchemy import and_, Date, cast
from flask_app import app
from models import *
import flask
from datetime import timedelta, datetime, timezone
from email.utils import parseaddr


@app.route('/api/v1/insert')
def insert_dummy_data():
    repo = Repositories(name='origin', org_name='openshift', url='github.com/openshift/origin')
    # We need to get repo_id dynamically here, because it can change.
    issue = Issues(number=4977, created_at='2019-04-13 10:22:23+00:00', updated_at='2019-04-13 10:22:23+00:00',
                   body=
                   "There seems to be no logging in the master logs even with loglevel=5 about "
                   "status 500 errors or general authentication errors when working "
                   "with anything other than AllowAll/DenyAll. In our case we were trying to setup "
                   "BasicAuth and continuously recieved 500 return code on the CLI end and just generic "
                   "error in the UI but nothing was logged in the master.This is happening on "
                   "OSE-3.0.2.0-0.git.16.4d626fe.el7ose.x86_64 "
                   "@liggitt has been looking into this with me for the last day or so.,",
                   title='Issue with auth not logging when it fails',
                   repo_id=1,
                   url='https://github.com/openshift/origin/issues/4977',
                   closed_at='2019-06-06 17:08:00+00:00',
                   is_security=1,
                   is_probable_cve=1,
                   is_passed_through_model=1)
    probable_cve = ProbableCVE(ecosystem='go', cause_type='issue', cause_id=4977, confidence_score=80.0,
                               identified_date='2019-10-07 02:34:00+00:00',
                               review_comments='Is a CVE. Good find.',
                               review_status=1,
                               is_cve=1,
                               is_triaged_internally=1,
                               last_updated_at='2019-10-07 02:34:00+00:00',
                               reviewed_by='aagam@redhat.com')
    db.session.add(repo)
    db.session.add(issue)
    db.session.add(probable_cve)
    db.session.commit()
    return flask.jsonify(response="Data inserted"), 200


@app.route('/api/v1/delete')
def delete_dummy_data():
    repo = Repositories.query.filter_by(url='github.com/openshift/origin').first_or_404(
        description='There is no repository to delete')
    issue = Issues.query.filter_by(url='https://github.com/openshift/origin/issues/4977').first_or_404(
        description='There is no issue to delete'
    )
    db.session.delete(issue)
    db.session.delete(repo)
    db.session.commit()
    return flask.jsonify(delete=True, message='Dummy records deleted'), 200


@app.route('/api/v1/getcve')
def get_cve():
    start_date = flask.request.args.get('start-date')
    end_date = flask.request.args.get('end-date')
    response = ProbableCVE.query.filter(and_(cast(ProbableCVE.identified_date, Date) <= end_date,
                                             cast(ProbableCVE.identified_date, Date) >= start_date)).first_or_404()
    return flask.jsonify(id=response.id, cve_id=response.cve_id,
                         ecosystem=response.ecosystem, cause_type=response.cause_type,
                         cause_id=response.cause_id, cve_date=response.cve_date,
                         confidence_score=response.confidence_score, identified_date=response.identified_date,
                         review_comments=response.review_comments, review_status=response.review_status,
                         is_cve=response.is_cve, is_triaged_internally=response.is_triaged_internally,
                         last_updated_at=response.last_updated_at, reviewed_by=response.reviewed_by), 200


def validate_request_data(input_json):
    """Validate the data.

    :param input_json: dict, describing data
    :return: boolean, result
    """
    validate_string = "{} cannot be empty"
    if 'id' not in input_json:
        validate_string = validate_string.format("id")
        return False, validate_string

    if 'is_cve' not in input_json:
        validate_string = validate_string.format("is_cve")
        return False, validate_string

    if 'reviewed_by' not in input_json:
        validate_string = validate_string.format("reviewed_by")
        return False, validate_string

    return True, None


@app.route('/api/v1/updatecve', methods=['POST'])
def update_cve():
    body = flask.request.get_json()
    validated_data = validate_request_data(body)
    if not validated_data[0]:
        return flask.jsonify(message=validated_data[1]), 400
    cve_id = body.get('id')
    is_cve = body.get('is_cve')
    reviewed_by = body.get('reviewed_by')
    is_valid_email = parseaddr(reviewed_by)
    if is_valid_email == ('', ''):
        return flask.jsonify(message='Please enter valid email id in the reviewed_by section'), 400
    probable_cve = ProbableCVE.query.get_or_404(cve_id, description='There is no data with id as {}'.format(cve_id))
    probable_cve.is_cve = is_cve
    review_comments = body.get('review_comments', probable_cve.review_comments)
    probable_cve.review_comments = review_comments
    probable_cve.reviewed_by = reviewed_by
    probable_cve.last_updated_at = datetime.now(tz=timezone(timedelta(), 'UTC')).isoformat(' ', 'seconds')
    db.session.commit()
    return flask.jsonify(message='CVEs updated'), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
