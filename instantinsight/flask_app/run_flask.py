#!/usr/bin/env python
"""Run the Flask application using Poetry."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask_app.app import app

if __name__ == "__main__":
    app.run(debug=True, port=5007)
