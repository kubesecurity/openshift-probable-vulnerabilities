"""Models for GO CVE DB."""

from flask_app import db


class Repositories(db.Model):
    __tablename__ = 'repositories'
    id = db.Column(db.Integer, primary_key=True)
    org_name = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(255), unique=True, nullable=False)


class Issues(db.Model):
    __tablename__ = 'issues'
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, nullable=False)
    updated_at = db.Column(db.TIMESTAMP, nullable=False)
    body = db.Column(db.Text)
    title = db.Column(db.Text)
    repo_id = db.Column(db.Integer, db.ForeignKey('repositories.id'), nullable=False)
    url = db.Column(db.String(255), unique=True, nullable=False)
    closed_at = db.Column(db.TIMESTAMP, nullable=False)
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)


class PullRequests(db.Model):
    __tablename__ = 'pullrequests'
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, nullable=False)
    updated_at = db.Column(db.TIMESTAMP, nullable=False)
    merged_at = db.Column(db.TIMESTAMP)
    closed_at = db.Column(db.TIMESTAMP)
    title = db.Column(db.Text, nullable=False)
    body = db.Column(db.Text)
    repo_id = db.Column(db.Integer, db.ForeignKey('repositories.id'), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    patch_url = db.Column(db.String(255), nullable=False)
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)


class Commits(db.Model):
    __tablename__ = 'commits'
    id = db.Column(db.Integer, primary_key=True)
    sha = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.TIMESTAMP)
    message = db.Column(db.Text, nullable=False)
    repo_id = db.Column(db.Integer, db.ForeignKey('repositories.id'), nullable=False)
    url = db.Column(db.String(255))
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)


class ProbableCVE(db.Model):
    __tablename__ = 'probablecves'
    id = db.Column(db.Integer, primary_key=True)
    cve_id = db.Column(db.String(255))
    ecosystem = db.Column(db.Enum('go', 'npm', 'maven', 'pypi', name='ecosystem'))
    cause_type = db.Column(db.Enum('issue', 'commit', 'pr', name='cause_type'))
    cause_id = db.Column(db.Integer, nullable=False)
    cve_date = db.Column(db.TIMESTAMP)
    confidence_score = db.Column(db.DECIMAL(5, 2))
    identified_date = db.Column(db.TIMESTAMP, index=True)
    review_comments = db.Column(db.Text)
    review_status = db.Column(db.Integer)
    is_cve = db.Column(db.Integer)
    is_triaged_internally = db.Column(db.Integer)
    last_updated_at = db.Column(db.TIMESTAMP)
    reviewed_by = db.Column(db.Text)

