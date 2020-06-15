"""Contains text normalization functions."""

import numpy as np
import contractions
from bs4 import BeautifulSoup
import unicodedata
import re
from concurrent import futures
import threading
import daiquiri
import logging

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


def strip_html_tags(text):
    """Strip html tags from the given text."""
    soup = BeautifulSoup(text, "lxml")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text, re.I)
    return stripped_text


def remove_urls(text):
    """Strip html tags from the given text."""
    url_pattern = (r'((https?:\/\/)(\s)*(www\.)?|(www\.))(\s)*((\w|\s)+\.)*'
                   r'([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*')
    text = re.sub(url_pattern, ' ', text, re.I)
    return text


def remove_checklists(text):
    """Strip html tags from the given text."""
    checklist_pattern = r'\[[xX\.\s]\]'
    text = re.sub(checklist_pattern, ' ', text, re.I | re.DOTALL)
    return text


def remove_accented_chars(text):
    """Strip html tags from the given text."""
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    """Strip html tags from the given text."""
    return contractions.fix(text)


def remove_special_characters(text, remove_digits=False):
    """Strip html tags from the given text."""
    pattern = r'[^a-zA-Z0-9/\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, ' ', text)
    return text


def pre_process_document(document):
    """Strip html tags from the given text."""
    # strip HTML
    document = strip_html_tags(document)

    # remove URLS
    document = remove_urls(document)

    # remove checklists
    document = remove_checklists(document)

    # expand contractions
    document = expand_contractions(document)

    # lower case
    document = document.lower()

    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))

    # remove accented characters
    document = remove_accented_chars(document)

    # remove special characters and\or digits
    # insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=False)

    # remove only numbers
    document = re.sub(r'\b\d+\b', '', document)

    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()

    return document


def pre_process_document_bert(document):
    """Strip html tags from the given text."""
    # strip HTML
    document = strip_html_tags(document)

    # remove URLS
    document = remove_urls(document)

    # remove checklists
    document = remove_checklists(document)

    # lower case
    document = document.lower()

    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))

    # remove accented characters
    document = remove_accented_chars(document)

    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()

    return document


def parallel_preprocessing(idx, doc, total_docs):
    """Strip html tags from the given text."""
    if idx % 5000 == 0 or idx == (total_docs - 1):
        _logger.info('%s working on doc num: %d',
                     threading.current_thread().name,
                     idx)
    return pre_process_document(doc)


def parallel_preprocessing_bert(idx, doc, total_docs):
    """Strip html tags from the given text."""
    if idx % 5000 == 0 or idx == (total_docs - 1):
        _logger.info('%s working on doc num: %d',
                     threading.current_thread().name,
                     idx)
    return pre_process_document_bert(doc)


def pre_process_documents_parallel(documents):
    """Strip html tags from the given text."""
    total_docs = len(documents)
    docs_input = [[idx, doc, total_docs] for idx, doc in enumerate(documents)]

    ex = futures.ThreadPoolExecutor(max_workers=None)
    _logger.info('Text Pre-processing: starting')
    norm_descriptions_map = ex.map(parallel_preprocessing,
                                   [record[0] for record in docs_input],
                                   [record[1] for record in docs_input],
                                   [record[2] for record in docs_input])
    norm_descriptions = list(norm_descriptions_map)
    return norm_descriptions


def pre_process_documents_parallel_bert(documents):
    """Strip html tags from the given text."""
    total_docs = len(documents)
    docs_input = [[idx, doc, total_docs] for idx, doc in enumerate(documents)]

    ex = futures.ThreadPoolExecutor(max_workers=None)
    _logger.info('Text Pre-processing: starting')
    norm_descriptions_map = ex.map(parallel_preprocessing_bert,
                                   [record[0] for record in docs_input],
                                   [record[1] for record in docs_input],
                                   [record[2] for record in docs_input])
    norm_descriptions = list(norm_descriptions_map)
    return norm_descriptions
