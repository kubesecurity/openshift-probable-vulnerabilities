#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

pushd "${SCRIPT_DIR}/.." > /dev/null

set -e
set -x

COVERAGE_THRESHOLD=75

export TERM=xterm

# set up terminal colors
RED=$(tput bold && tput setaf 1)
GREEN=$(tput bold && tput setaf 2)
YELLOW=$(tput bold && tput setaf 3)
NORMAL=$(tput sgr0)


echo "Create Virtualenv for Python deps ..."

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

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
    python3 "$(which pip3)" install -r requirements.txt

}

check_python_version

[ "$NOVENV" == "1" ] || prepare_venv || exit 1

$(which pip3) install -r requirements-test.txt

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=`pwd` python3 "$(which pytest)" --cov=src/ --cov-report term-missing --cov-fail-under=$COVERAGE_THRESHOLD -vv tests/

codecov --token=4c49034f-36ec-4e2c-b839-6e0f615aaebe
printf "%stests passed%s\n\n" "${GREEN}" "${NORMAL}"


popd > /dev/null
