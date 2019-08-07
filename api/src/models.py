"""Models for GO CVE DB."""

from flask_app import db
from sqlalchemy.orm import validates
from sqlalchemy.schema import CheckConstraint
from urllib.parse import urlparse
from datetime import timedelta, datetime, timezone


def get_timestamp():
    return datetime.now(tz=timezone(timedelta(), 'UTC')).isoformat(' ', 'seconds')


class Repositories(db.Model):
    __tablename__ = 'repositories'
    id = db.Column(db.Integer, primary_key=True)
    org_name = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(255), unique=True, nullable=False)

    @validates('url')
    def validate_url(self, key, url):
        o = urlparse(url)
        assert o.geturl(), "Please provide valid url"
        return url


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
    closed_at = db.Column(db.TIMESTAMP)
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)

    __table_args__ = (
        CheckConstraint('updated_at >= created_at'),
    )

    @validates('url')
    def validate_url(self, key, url):
        o = urlparse(url)
        assert o.geturl(), "Please provide valid url"
        return url

    @validates('is_security', 'is_probable_cve', 'is_passed_through_model')
    def validate_flags(self, key, flag):
        assert int(flag) in (0, 1), "Please provide flag as 0 or 1"
        return flag


class PullRequests(db.Model):
    __tablename__ = 'pullrequests'
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, nullable=False)
    updated_at = db.Column(db.TIMESTAMP, nullable=False)
    merged_at = db.Column(db.TIMESTAMP)
    closed_at = db.Column(db.TIMESTAMP)
    title = db.Column(db.Text)
    body = db.Column(db.Text)
    repo_id = db.Column(db.Integer, db.ForeignKey('repositories.id'), nullable=False)
    url = db.Column(db.String(255), unique=True, nullable=False)
    patch_url = db.Column(db.String(255), unique=True, nullable=False)
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)

    __table_args__ = (
        CheckConstraint('updated_at >= created_at'),
    )

    @validates('url')
    def validate_url(self, key, url):
        o = urlparse(url)
        assert o.geturl(), "Please provide valid url"
        return url

    @validates('is_security', 'is_probable_cve', 'is_passed_through_model')
    def validate_flags(self, key, flag):
        assert int(flag) in (0, 1), "Please provide flag as 0 or 1"
        return flag


class Commits(db.Model):
    __tablename__ = 'commits'
    id = db.Column(db.Integer, primary_key=True)
    sha = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.TIMESTAMP)
    message = db.Column(db.Text, nullable=False)
    repo_id = db.Column(db.Integer, db.ForeignKey('repositories.id'), nullable=False)
    url = db.Column(db.String(255), unique=True, nullable=False)
    is_security = db.Column(db.Integer)
    is_probable_cve = db.Column(db.Integer)
    is_passed_through_model = db.Column(db.Integer)

    @validates('url')
    def validate_url(self, key, url):
        o = urlparse(url)
        assert o.geturl(), "Please provide valid url"
        return url

    @validates('is_security', 'is_probable_cve', 'is_passed_through_model')
    def validate_flags(self, key, flag):
        assert int(flag) in (0, 1), "Please provide flag as 0 or 1"
        return flag


class ProbableCVE(db.Model):
    __tablename__ = 'probablecves'
    id = db.Column(db.Integer, primary_key=True)
    cve_id = db.Column(db.String(255))
    ecosystem = db.Column(db.Enum('go', 'npm', 'maven', 'pypi', name='ecosystem'))
    cause_type = db.Column(db.Enum('issue', 'commit', 'pr', name='cause_type'))
    cause_id = db.Column(db.Integer, nullable=False)
    cve_date = db.Column(db.TIMESTAMP)
    confidence_score = db.Column(db.DECIMAL(5, 2))
    identified_date = db.Column(db.TIMESTAMP, index=True, nullable=False)
    review_comments = db.Column(db.Text)
    review_status = db.Column(db.Integer)
    is_cve = db.Column(db.Integer)
    is_triaged_internally = db.Column(db.Integer)
    last_updated_at = db.Column(db.TIMESTAMP, nullable=False, onupdate=get_timestamp,
                                default=get_timestamp)
    reviewed_by = db.Column(db.Text)

    __table_args__ = (
        CheckConstraint('last_updated_at >= identified_date'),
    )

    @validates('review_status', 'is_cve', 'is_triaged_internally')
    def validate_flags(self, key, flag):
        assert int(flag) in (0, 1), "Please provide flag as 0 or 1"
        return flag

    @validates('reviewed_by')
    def validate_email(self, key, address):
        assert '@' in address, "Please provide valid email id"
        return address

