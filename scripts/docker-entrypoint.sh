#!/bin/bash
# docker-entrypoint.sh - Entry point for AlphaZero Docker container

set -e

# Run the command if it starts with a hyphen or if it's not a path
if [ "${1:0:1}" = '-' ] || ! [ -x "$1" ]; then
    exec /app/build/bin/alphazero_cli "$@"
else
    # Otherwise, execute the command directly
    exec "$@"
fi