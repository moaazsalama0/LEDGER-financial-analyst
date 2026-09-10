#!/bin/sh
set -eu

old_ifs=$IFS
IFS=:
set -- ${REQUIREMENTS_FILES:-/app/requirements.txt} -- "$@"
IFS=$old_ifs

requirements=""
while [ "$1" != "--" ]; do
    requirements="$requirements $1"
    shift
done
shift

# This is intentionally offline: dependencies are baked into the image.
python /opt/ledger/check_requirements.py $requirements
python -m pip check

exec "$@"
