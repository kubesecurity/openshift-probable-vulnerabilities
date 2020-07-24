#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

pushd "${SCRIPT_DIR}/.." > /dev/null

set -e
set -x

COVERAGE_THRESHOLD=30

export TERM=xterm

# set up terminal colors
RED=$(tput bold && tput setaf 1)
GREEN=$(tput bold && tput setaf 2)
YELLOW=$(tput bold && tput setaf 3)
NORMAL=$(tput sgr0)


echo "Create Virtualenv for Python deps ..."

function prepare_venv() {
    VIRTUALENV=$(which virtualenv)
    if [ $? -eq 1 ]
    then
        # python34 which is in CentOS does not have virtualenv binary
        VIRTUALENV=$(which virtualenv-3)
    fi

    ${VIRTUALENV} -p python3 venv && source venv/bin/activate
    if [ $? -ne 0 ]
    then
        printf "%sPython virtual environment can't be initialized%s" "${RED}" "${NORMAL}"
        exit 1
    fi
    pip install -U pip
    python3 "$(which pip3)" install -r model_inference_triage_pipeline/requirements.txt

}

./qa/check_python_version.sh

[ "$NOVENV" == "1" ] || prepare_venv || exit 1

# $(which pip3) install -r requirements-test.txt
pip3 install pytest
pip3 install pytest-cov
pip3 install radon
pip3 install codecov

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=`pwd` python3 "$(which pytest)" --cov=model_inference_triage_pipeline --cov=utils --cov=tests --cov-report term-missing --cov-fail-under=$COVERAGE_THRESHOLD -vv tests

#codecov --token=72315105-7ad8-42c3-965e-64bb328e747a
printf "%stests passed%s\n\n" "${GREEN}" "${NORMAL}"


popd > /dev/null
