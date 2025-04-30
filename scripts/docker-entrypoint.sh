#!/bin/bash
set -e

# Handle test command
if [[ "$1" == "test" || "$1" == "tests" ]]; then
  exec /app/run-tests.sh
# Handle Python test command
elif [[ "$1" == "pytest" ]]; then
  exec python3 -m pytest /app/python/tests
# Default to CLI if command starts with dash or is not executable
elif [[ "${1:0:1}" == "-" || ! -x "$1" ]]; then
  if [ -x /app/bin/alphazero_cli ]; then
    exec /app/bin/alphazero_cli "$@"
  else
    echo "ERROR: alphazero_cli not found"
    exit 1
  fi
# Otherwise execute the command directly
else
  exec "$@"
fi