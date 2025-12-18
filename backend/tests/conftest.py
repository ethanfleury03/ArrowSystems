import os


def pytest_configure():
    """
    Test suite bootstrap.

    backend.config.env instantiates Settings() at import time, and Settings requires DATABASE_URL.
    In CI this may already be present, but for local runs we provide a safe default.
    """
    os.environ.setdefault("ENV", "dev")
    os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    # Some unit tests flip ENV=prod and reload Settings; provide a non-empty default secret
    # so tests don't depend on external environment configuration.
    os.environ.setdefault("FRONTEND_SESSION_SECRET", "dev-frontend-session-secret-for-tests-only")


