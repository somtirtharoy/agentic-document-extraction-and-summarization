import os

import pytest

# Must be set before any module that calls get_settings() is imported.
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCS_BUCKET", "test-bucket")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
