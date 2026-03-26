from __future__ import annotations

import grpc

from ..retry import RetryPolicy, call_with_retry


def init_worker_with_retry(
    stub,
    slice_cfg,
    *,
    address: str,
    policy: RetryPolicy,
    logger,
    phase: str,
):
    def _call():
        response = stub.init(slice_cfg, timeout=policy.rpc_timeout_s)
        if not response.ok:
            raise RuntimeError(response.message or f"{phase} rejected by {address}")
        return response

    def _should_retry(exc: Exception) -> bool:
        return isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.UNAVAILABLE

    def _on_retry(attempt: int, max_attempts: int, exc: Exception, delay_s: float):
        logger.warning(
            "%s retry %s/%s for %s failed: %s; next_delay=%.1fs",
            phase,
            attempt,
            max_attempts,
            address,
            exc,
            delay_s,
        )

    try:
        return call_with_retry(
            _call,
            policy=policy,
            should_retry=_should_retry,
            on_retry=_on_retry,
        )
    except Exception as exc:
        raise RuntimeError(f"{phase} failed for worker {address}: {exc}") from exc
