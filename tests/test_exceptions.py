from omniai.exceptions import (
    OmniaiError,
    OmniaiAuthError,
    OmniaiPaymentRequiredError,
    OmniaiNotFoundError,
    OmniaiRateLimitError,
    OmniaiExecutionError,
    OmniaiTimeoutError,
    OmniaiConnectionError,
    OmniaiValidationError,
    OmniaiUnsupportedProviderError,
    OmniError,
    OmniAuthError,
    OmniExecutionError,
    OmniTimeoutError,
    OmniRateLimitError,
)


def test_exception_inheritance():
    assert issubclass(OmniaiAuthError, OmniaiError)
    assert issubclass(OmniaiPaymentRequiredError, OmniaiError)
    assert issubclass(OmniaiNotFoundError, OmniaiError)
    assert issubclass(OmniaiRateLimitError, OmniaiError)
    assert issubclass(OmniaiExecutionError, OmniaiError)
    assert issubclass(OmniaiTimeoutError, OmniaiError)
    assert issubclass(OmniaiConnectionError, OmniaiError)
    assert issubclass(OmniaiValidationError, OmniaiError)
    assert issubclass(OmniaiUnsupportedProviderError, OmniaiError)


def test_backward_compatible_aliases():
    assert OmniError is OmniaiError
    assert OmniAuthError is OmniaiAuthError
    assert OmniExecutionError is OmniaiExecutionError
    assert OmniTimeoutError is OmniaiTimeoutError
    assert OmniRateLimitError is OmniaiRateLimitError


def test_rate_limit_retry_after():
    err = OmniaiRateLimitError("Too many requests", retry_after=12.5)
    assert err.retry_after == 12.5
    assert str(err) == "Too many requests"


def test_execution_error_status_code():
    err = OmniaiExecutionError("Upstream model error", status_code=502, details={"provider": "openai"})
    assert err.status_code == 502
    assert err.details["provider"] == "openai"


def test_validation_error_raw_output():
    err = OmniaiValidationError("Invalid JSON", raw_output="not-json")
    assert err.raw_output == "not-json"
