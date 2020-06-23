#!/bin/bash

echo "import sys; assert(sys.version_info.major, sys.version_info.minor)>=(int(sys.argv[1]), int(sys.argv[2])), sys.version_info" | python3 - 3 6
