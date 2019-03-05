#!/bin/sh


EXIT_ON_FAILURE=0
for i in "$@"
do
    if [[ $i == "-"*"x"* ]]; then
        EXIT_ON_FAILURE=1
    fi
done

PYTHONPATH=../opt/common/python:../opt/imfil/python:../opt/snobfit/python:$PYTHONPATH

pytest $@ test*.py
if [ $? -eq 0 ]; then
   pytest $@ ../opt/imfil/test/test*.py
elif [ ${EXIT_ON_FAILURE} -eq 1 ]; then
   exit $?
fi
if [ $? -eq 0 ]; then
   pytest $@ ../opt/snobfit/test/test*.py
fi
