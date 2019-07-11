import json

from jira import JIRA
import boto3
import os

# Comment out below code for parallelism. It can fail because of pickling issues.

# def get_issues_from_jira_api(project_string, jira_obj):
#     if not hasattr(jira_obj._session, 'max_retries'):
#         setattr(jira_obj._session, 'max_retries', 5)
#     if not hasattr(jira_obj._session, 'timeout'):
#         setattr(jira_obj._session, 'timeout', 300)
#     print("Project:{}".format(project_string) + " Started")
#     try:
#         issues = jira_obj.search_issues(jql_str="project={}".format(project_string), maxResults=False, fields='*all')
#         print("Project:{}".format(project_string) + " Completed")
#         return issues
#     except Exception:
#         print("Exception in project: {}".format(project_string))
#         return []


if __name__ == '__main__':
    project_names_set = set()

    jac = JIRA(server='http://www.bouncycastle.org/jira')

    bucket_resource = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_S3_ACCESS_KEY_ID'),
                                   aws_secret_access_key=os.environ.get('AWS_S3_SECRET_ACCESS_KEY'),
                                   region_name='us-east-1')

    with open('project_names.txt', 'r') as f:
        for l in f.readlines():
            project_names_set.add(l.split('/')[-1].split('-')[0])

    # Comment out this code to fetch issues in parallel.
    # This can fail though because of pickling issues.
    # See more at: https://github.com/pycontribs/jira/issues/306.

    # manager = Manager()
    # issue_raw_list = manager.list()
    #
    # with Pool() as pool:
    #     results = pool.map(partial(get_issues_from_jira_api, jira_obj=jac), project_names_set)
    #     for r in results:
    #         issue_raw_list.append(r.raw)

    issue_raw_list = []

    for idx, p in enumerate(project_names_set):
        project_string = "project={}".format(p)
        try:
            # Set maxResults=False to get all the issues and fields=*all for all fields.
            issue = jac.search_issues(jql_str=project_string, maxResults=False, fields='*all')
            for i in issue:
                issue_raw_list.append(i.raw)
            print("Project number {} with name {} completed".format(idx, p))
        except Exception as e:
            print("Exception triggered for project {}".format(p))
            print("Exception is : {}".format(e))

    with open('jira_bouncycastle_issues.json', 'w') as f:
        print("Dumping into JSON")
        json.dump(issue_raw_list, f)

    bucket_resource.upload_file('jira_bouncycastle_issues.json', 'aagshah-node-package-data',
                                'jira_bouncycastle_issues.json')
