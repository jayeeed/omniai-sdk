import httpx
import pytest
from omniai._base import RetryPolicy


def test_retry_policy_defaults():
    policy = RetryPolicy()
    assert policy.max_retries == 3
    assert policy.backoff_factor == 0.5
    assert 429 in policy.retryable_statuses
    assert 500 in policy.retryable_statuses
    assert 503 in policy.retryable_statuses


def test_retry_after_parsing():
    policy = RetryPolicy()

    # Float/integer seconds in header
    resp = httpx.Response(429, headers={"retry-after": "5"})
    assert policy.parse_retry_after(resp) == 5.0

    # Float with decimal
    resp2 = httpx.Response(429, headers={"retry-after": "2.5"})
    assert policy.parse_retry_after(resp2) == 2.5

    # Missing header
    resp3 = httpx.Response(429, headers={})
    assert policy.parse_retry_after(resp3) is None


def test_should_retry_logic():
    policy = RetryPolicy(max_retries=3)

    # Attempt 1 on 429 -> True
    resp_429 = httpx.Response(429)
    assert policy.should_retry(1, response=resp_429) is True

    # Attempt 1 on 500 -> True
    resp_500 = httpx.Response(500)
    assert policy.should_retry(1, response=resp_500) is True

    # Attempt 1 on 401 (Auth error) -> False (Client error must not be retried)
    resp_401 = httpx.Response(401)
    assert policy.should_retry(1, response=resp_401) is False

    # Attempt 1 on 402 (Payment required) -> False
    resp_402 = httpx.Response(402)
    assert policy.should_retry(1, response=resp_402) is False

    # Attempt 3 (equal to max_retries) -> False (exceeded)
    assert policy.should_retry(3, response=resp_500) is False

    # Connection error -> True
    conn_err = httpx.ConnectError("Connection refused")
    assert policy.should_retry(1, exception=conn_err) is True


def test_execute_sync_retries_and_succeeds():
    policy = RetryPolicy(max_retries=3, backoff_factor=0.01)

    calls = 0

    def mock_request():
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, json={"result": "success"})

    resp = policy.execute_sync(mock_request, operation_name="test_call")
    assert resp.status_code == 200
    assert resp.json() == {"result": "success"}
    assert calls == 2


def test_execute_sync_does_not_retry_auth_error():
    policy = RetryPolicy(max_retries=3, backoff_factor=0.01)

    calls = 0

    def mock_request():
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"detail": "Unauthorized"})

    resp = policy.execute_sync(mock_request, operation_name="test_call")
    assert resp.status_code == 401
    assert calls == 1


@pytest.mark.asyncio
async def test_execute_async_retries_and_succeeds():
    policy = RetryPolicy(max_retries=3, backoff_factor=0.01)

    calls = 0

    async def mock_async_request():
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, headers={"retry-after": "0.01"}, text="Too Many Requests")
        return httpx.Response(200, json={"result": "async success"})

    resp = await policy.execute_async(mock_async_request, operation_name="test_async_call")
    assert resp.status_code == 200
    assert resp.json() == {"result": "async success"}
    assert calls == 3
