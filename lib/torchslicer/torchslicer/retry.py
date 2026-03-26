import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    delay_s: float = 0.0
    rpc_timeout_s: float = 5.0


def call_with_retry(
    func: Callable[[], object],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[int, int, Exception, float], None] | None = None,
):
    last_exc = None
    for attempt in range(1, max(1, policy.max_attempts) + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            retryable = should_retry(exc) if should_retry is not None else True
            if attempt >= max(1, policy.max_attempts) or not retryable:
                raise
            if on_retry is not None:
                on_retry(attempt, policy.max_attempts, exc, policy.delay_s)
            if policy.delay_s > 0:
                time.sleep(policy.delay_s)
    raise last_exc
