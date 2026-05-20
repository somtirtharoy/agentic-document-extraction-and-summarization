from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Reusable decorator for GCP API calls that can hit quota limits or transient errors.
gcp_retry = retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
