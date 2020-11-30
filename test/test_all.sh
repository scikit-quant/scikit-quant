#!/bin/sh


EXIT_ON_FAILURE=0
for i in "$@"
do
    if [[ $i == "-"*"x"* ]]; then
        EXIT_ON_FAILURE=1
    fi
done

PYTHONPATH=../opt/common/python:../opt/imfil/python:../opt/snobfit/python:../opt/nomad/python:$PYTHONPATH

pytest $@ test*.py
if [ $? -ne 0 ]; then
  if [ ${EXIT_ON_FAILURE} -eq 1 ]; then
    exit $?
  fi
fi
pytest $@ ../opt/imfil/test/test*.py
pytest $@ ../opt/snobfit/test/test*.py
pytest $@ ../opt/nomad/test/test*.py
