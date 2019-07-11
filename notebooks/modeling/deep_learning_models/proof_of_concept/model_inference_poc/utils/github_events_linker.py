import re
import numpy as np
import os
import requests
import time


def link_github_events(url, event_type, github_user, github_auth):
    def get_repo_name(url):
        '''
          Helps in getting the repository name which is useful
          for querying in GitHub API queries.
        '''
        pattern = re.compile(r'.*?github.com/(.*)/(.*)/.*', re.I)
        repo_name = pattern.search(url).group(1)
        return repo_name

    def get_commits_changed_files(patch_urls, repo_url):
        '''
          Helps in getting the relevant commits and files changed
          from a list of GitHub patch URLs for a specific repository
        '''
        commits = []
        changed_files = []

        for patch_url in patch_urls:
            response = requests.get(patch_url)
            data = response.text
            if data:
                commit_hashes = re.findall(r'(?:\n|^)from (.*?)\s', data, re.I)
                if commit_hashes:
                    commit_hashes = [item for item in commit_hashes if item.isalnum()]
                    commit_links = [repo_url + '/commit/' + item.rstrip('/') for item in commit_hashes]
                    commits.extend(commit_links)

                changed_file_paths = re.findall(r"(?:\n|^)diff\s--git\sa.*?\.go\sb(.*\.go)", data, re.I)
                if changed_file_paths:
                    changed_file_paths = [item.rstrip('/') for item in changed_file_paths]
                    changed_files.append([patch_url] + list(np.unique(changed_file_paths)))

        commits = list(np.unique(commits))
        return commits, changed_files

    def get_dependent_links_from_issue(url, repo_name, repo_url, requests_made):
        '''
          Helps in getting the relevant and related PRs, commits
          and files changed based on a GitHub issue,
          for a specfic repository
        '''
        issue_urls = [url]
        pr_urls = []
        patch_urls = []
        commit_urls = []
        files_changed = []

        issue_num_pattern = re.compile('https://github.com/.*/issues/(.*)', re.I)
        issue_num = issue_num_pattern.search(url).group(1)

        pr_search_url = 'https://api.github.com/search/issues?q=is:pr issue:{issue_num} repo:{repo_name}'
        pr_search_url = pr_search_url.format(issue_num=issue_num, repo_name=repo_name)
        response = requests.get(pr_search_url, auth=(github_user, github_auth))
        requests_made += 1
        if response.status_code == 200:
            content = response.json()
            if content and content.get('items'):

                pr_details = list(filter(None, [record.get('pull_request')
                                                for record in content['items']]))
                pr_urls = list(filter(None, [record.get('html_url')
                                             for record in pr_details]))
                if pr_details:
                    patch_urls = list(filter(None, [record.get('patch_url')
                                                    for record in pr_details]))
                    commit_urls, files_changed = get_commits_changed_files(patch_urls, repo_url)

        return issue_urls, pr_urls, commit_urls, files_changed, requests_made

    def get_dependent_links_from_pr(url, repo_name, repo_url, requests_made):
        '''
          Helps in getting the relevant and related issues, commits
          and files changed based on a GitHub pull request (PR),
          for a specfic repository
        '''
        issue_urls = []
        pr_urls = [url]
        patch_urls = []
        commit_urls = []
        files_changed = []

        pr_num_pattern = re.compile('https://github.com/.*/pull/(.*)', re.I)
        pr_num = pr_num_pattern.search(url).group(1)

        issue_search_url = 'https://api.github.com/search/issues?q=is:issue pr:{pr_num} repo:{repo_name}'
        issue_search_url = issue_search_url.format(pr_num=pr_num, repo_name=repo_name)
        response = requests.get(issue_search_url, auth=(github_user, github_auth))
        requests_made += 1
        if response.status_code == 200:
            content = response.json()
            if content and content.get('items'):
                issue_urls = list(filter(None, [record.get('html_url')
                                                for record in content['items']]))

        patch_urls = [record + '.patch' for record in pr_urls]
        commit_urls, files_changed = get_commits_changed_files(patch_urls, repo_url)

        return issue_urls, pr_urls, commit_urls, files_changed, requests_made

    def get_dependent_links_from_commit(url, repo_name, repo_url, requests_made):
        '''
          Helps in getting the relevant and related issues, PRs
          and files changed based on a GitHub commit,
          for a specfic repository
        '''
        issue_urls = []
        pr_urls = []
        patch_urls = []
        commit_urls = [url]
        files_changed = []

        commit_num_pattern = re.compile('https://github.com/.*/commit/(.*)', re.I)
        commit_num = commit_num_pattern.search(url).group(1)

        pr_search_url = 'https://api.github.com/search/issues?q=is:pr commit:{commit_num} repo:{repo_name}'
        pr_search_url = pr_search_url.format(commit_num=commit_num, repo_name=repo_name)
        response = requests.get(pr_search_url, auth=(github_user, github_auth))
        requests_made += 1
        if response.status_code == 200:
            content = response.json()
            if content and content.get('items'):
                pr_urls = list(filter(None, [record.get('html_url')
                                             for record in content['items']]))
                if pr_urls:
                    patch_urls = [record + '.patch' for record in pr_urls]
                    commit_urls, files_changed = get_commits_changed_files(patch_urls, repo_url)

        pr_num_pattern = re.compile('https://github.com/.*/pull/(.*)', re.I)
        for pr_url in pr_urls:
            pr_num = pr_num_pattern.search(pr_url).group(1)
            issue_search_url = 'https://api.github.com/search/issues?q=is:issue pr:{pr_num} repo:{repo_name}'
            issue_search_url = issue_search_url.format(pr_num=pr_num, repo_name=repo_name)
            response = requests.get(issue_search_url, auth=(github_user, github_auth))
            requests_made += 1
            if response.status_code == 200:
                content = response.json()
                if content and content.get('items'):
                    issue_urls.extend(list(filter(None, [record.get('html_url')
                                                         for record in content['items']])))

        return issue_urls, pr_urls, commit_urls, files_changed, requests_made

    try:
        issue_urls = []
        pr_urls = []
        patch_urls = []
        commit_urls = []
        files_changed = []
        requests_made = 0
        repo_name = get_repo_name(url)
        repo_url = 'https://github.com/' + repo_name

        if event_type.lower() == 'issue':
            issue_urls, pr_urls, \
            commit_urls, files_changed, requests_made = get_dependent_links_from_issue(url, repo_name,
                                                                                       repo_url,
                                                                                       requests_made)
        elif event_type.lower() == 'pull request':
            issue_urls, pr_urls, \
            commit_urls, files_changed, requests_made = get_dependent_links_from_pr(url, repo_name,
                                                                                    repo_url,
                                                                                    requests_made)
        elif event_type.lower() == 'commit':
            issue_urls, pr_urls, \
            commit_urls, files_changed, requests_made = get_dependent_links_from_commit(url, repo_name,
                                                                                        repo_url,
                                                                                        requests_made)
    except Exception as e:
        print(repr(e))  # TODO: logging in the future

    return ({
                'issue_url': issue_urls,
                'fixed_url': pr_urls,
                'commit_url': commit_urls,
                'files_changed': files_changed
            }, requests_made)


def generate_github_events_dependency_data(gh_urls, gh_event_types, github_user, github_auth):
    total_requests_made = 0
    events_link_data = []
    # TODO: needs to be handled better
    # what if there is 10+ URLs in one function call itself
    # (though probability is less, still need to be handled)
    for gh_url, gh_event_type in zip(gh_urls, gh_event_types):
        if total_requests_made >= 25:
            print("Total Requests Made: {} Sleeping for a minute".format(total_requests_made))
            time.sleep(65)
            total_requests_made = 0
        data, requests_made = link_github_events(gh_url, gh_event_type, github_user, github_auth)
        events_link_data.append(data)
        total_requests_made += requests_made

    return events_link_data


