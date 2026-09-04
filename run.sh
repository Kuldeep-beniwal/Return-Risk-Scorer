#!/bin/bash
# Double-click this (or run `./run.sh`) to start the Return-Risk Console.
# It activates the virtual environment (if present) and starts the server.

cd "$(dirname "$0")/backend"

if [ -d "../venv" ]; then
  source ../venv/bin/activate
fi

python3 app.py
