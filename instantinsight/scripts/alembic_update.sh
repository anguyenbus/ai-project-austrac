#!/bin/bash

# Run the alembic upgrade command
poetry run alembic upgrade head

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Alembic upgrade completed successfully"
else
    echo "Alembic upgrade failed"
    exit 1
fi