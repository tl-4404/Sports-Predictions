"""Compatibility entrypoint for the shared web application."""
import sys
from shared.web import app as application

if __name__ == "__main__":
    application.main()
else:
    # Keep existing Flask imports and test configuration on the actual module.
    sys.modules[__name__] = application
