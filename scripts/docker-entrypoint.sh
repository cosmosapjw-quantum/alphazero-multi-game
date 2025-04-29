#!/bin/bash
# docker-entrypoint.sh - Entry point for AlphaZero Docker container

set -e

# Special handling for ctest command
if [[ "$1" == "ctest" ]]; then
    exec "$@"
# Special handling for test command to run all tests
elif [[ "$1" == "test" ]] || [[ "$1" == "tests" ]] || [[ "$1" == "run-tests" ]]; then
    shift
    exec /app/run-tests.sh "$@"
# Special handling for debug-tests command
elif [[ "$1" == "debug-tests" ]]; then
    exec /app/debug-tests.sh
# Special handling for fix-ctest command
elif [[ "$1" == "fix-ctest" ]]; then
    exec /app/fix-ctest.sh
# Special handling for fix-tests command
elif [[ "$1" == "fix-tests" ]]; then
    exec /app/fix-tests-completely.sh
# Special handling for simple-test-fix command
elif [[ "$1" == "simple-test-fix" ]]; then
    exec /app/simple-test-fix.sh
# Run the command if it starts with a hyphen or if it's not a path
elif [ "${1:0:1}" = '-' ] || ! [ -x "$1" ]; then
    exec /app/bin/alphazero_cli "$@"
else
    # Otherwise, execute the command directly
    exec "$@"
fi