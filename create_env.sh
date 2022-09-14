#!/bin/bash

d=$(dirname "$0")

# Exit any virtual environments.
deactivate 2>/dev/null

# Create the virtual environment to execute the program.
python3 -m venv --symlinks $d/env

# Install the packages required by the program in the virtual environment.
. $d/env/bin/activate
$d/env/bin/python3 -m pip install -r $d/requirements.txt
deactivate