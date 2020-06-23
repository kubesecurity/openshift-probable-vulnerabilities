#!/bin/bash

set -ex

. cico_setup.sh

build_image

./qa/runtest.sh

push_image
