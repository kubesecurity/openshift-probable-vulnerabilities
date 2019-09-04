from utils import bq_utils as bqu
from utils import cloud_constants as cc
import os
import re
import numpy as np
import daiquiri
import logging

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


def create_github_bq_client():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cc.BIGQUERY_CREDENTIALS_FILEPATH
    gh_archive = bqu.BigQueryHelper(active_project="githubarchive",
                                    dataset_name="day")
    _logger.info('Setting up BQ Client: {client}'.format(client=gh_archive))
    return gh_archive


def get_gokube_trackable_repos(repo_dir):
    gh_repo_links = open(repo_dir).readlines()
    gh_repo_links = np.array([item.strip('\n').strip()
                              for item in gh_repo_links])

    pattern = re.compile(r'.*?github.com/(.*)', re.I)
    repo_names = np.array(list(filter(None, [pattern.search(item).group(1)
                                             if pattern.search(item) else None
                                             for item in gh_repo_links])))
    _logger.info('Total Repos to Track: {repos}'.format(repos=len(repo_names)))
    return repo_names


def bq_add_query_params(query, params_dict):
    for i, j in params_dict.items():
        query = query.replace(i, j)
    return query
