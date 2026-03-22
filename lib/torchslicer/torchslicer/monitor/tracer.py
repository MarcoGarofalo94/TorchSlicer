"""
Thin OpenTelemetry wrapper for TorchSlicer.

Design contract
---------------
- All public functions are safe to call even if `opentelemetry` is not installed.
- `span()` yields None when tracing is disabled — callers must guard with `if s:`.
- `BatchSpanProcessor` exports asynchronously; a lost Jaeger connection never
  blocks or raises in the training thread.
- Training exceptions always propagate; only OTEL-internal failures are swallowed.

Quick start (in coordinator / worker entrypoint)::

    from torchslicer.monitor import tracer
    tracer.configure()          # reads OTEL_EXPORTER_OTLP_ENDPOINT from env
    # … or explicitly:
    tracer.configure("http://jaeger:4317", service_name="torchslicer-coordinator")

If OTEL_EXPORTER_OTLP_ENDPOINT is not set and configure() is not called,
every span() call is a zero-overhead no-op.
"""

import os
import time
from contextlib import contextmanager

# ── optional OTEL import ───────────────────────────────────────────────────────

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    _OTEL_AVAILABLE = True
except ImportError:
    pass

# ── optional gRPC auto-instrumentation (propagates context across workers) ─────
_GRPC_INSTRUMENTED = False
try:
    from opentelemetry.instrumentation.grpc import (
        GrpcInstrumentorClient,
        GrpcInstrumentorServer,
    )
    _HAS_GRPC_INSTRUMENTATION = True
except ImportError:
    _HAS_GRPC_INSTRUMENTATION = False

# ── module state ──────────────────────────────────────────────────────────────

_tracer = None  # None → all spans are no-ops


# ── public API ────────────────────────────────────────────────────────────────

def configure(
    endpoint: str | None = None,
    service_name: str = "torchslicer",
    instrument_grpc: bool = True,
) -> None:
    """
    Set up the OTEL tracer and (optionally) auto-instrument gRPC channels.

    Parameters
    ----------
    endpoint:
        OTLP gRPC endpoint, e.g. ``"http://jaeger:4317"``.
        Defaults to the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var, or
        ``"http://localhost:4317"`` if neither is set.
    service_name:
        Appears as the service label in Jaeger.  Use different names for
        coordinator vs worker so they show up as separate services.
    instrument_grpc:
        If True and ``opentelemetry-instrumentation-grpc`` is installed,
        auto-instrument all gRPC clients/servers so trace context is
        propagated across coordinator ↔ worker calls automatically.
    """
    global _tracer, _GRPC_INSTRUMENTED

    if not _OTEL_AVAILABLE:
        return  # silent no-op

    try:
        endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _otel_trace.set_tracer_provider(provider)
        _tracer = _otel_trace.get_tracer("torchslicer")
    except Exception as exc:
        # Don't let a misconfigured endpoint crash startup
        import warnings
        warnings.warn(f"[torchslicer] tracing setup failed: {exc}", stacklevel=2)
        _tracer = None
        return

    if instrument_grpc and _HAS_GRPC_INSTRUMENTATION and not _GRPC_INSTRUMENTED:
        try:
            GrpcInstrumentorClient().instrument()
            GrpcInstrumentorServer().instrument()
            _GRPC_INSTRUMENTED = True
        except Exception:
            pass  # gRPC instrumentation is best-effort


def is_enabled() -> bool:
    return _tracer is not None


@contextmanager
def span(name: str, **attrs):
    """
    Context manager for a traced span.

    Usage::

        with tracer.span("worker.forward", batch_id=42, shape="(32,64)") as s:
            output = layer(input)
            if s:
                s.set_attribute("output_shape", str(output.shape))

    Yields the OTEL span object, or ``None`` if tracing is disabled.
    The caller should always guard attribute access with ``if s:``.
    """
    if _tracer is None:
        yield None
        return

    t0 = time.perf_counter()
    _training_exc = None

    try:
        with _tracer.start_as_current_span(name) as s:
            # Set initial attributes
            for k, v in attrs.items():
                _safe_set(s, k, v)

            try:
                yield s
            except Exception as exc:
                _training_exc = exc
                try:
                    s.record_exception(exc)
                except Exception:
                    pass
                # Don't re-raise yet — let the span close cleanly first

            # Duration is set after yield returns (or after training exc is caught)
            try:
                s.set_attribute("duration_ms", round((time.perf_counter() - t0) * 1000, 2))
            except Exception:
                pass

    except Exception:
        if _training_exc is None:
            # OTEL internal failure — swallow silently, yield already happened
            pass

    if _training_exc is not None:
        raise _training_exc


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_set(s, key: str, value) -> None:
    """Set a span attribute, coercing to a supported type. Never raises."""
    if s is None:
        return
    try:
        if isinstance(value, (bool, int, float, str)):
            s.set_attribute(key, value)
        else:
            s.set_attribute(key, str(value))
    except Exception:
        pass


def auto_configure_if_env() -> None:
    """
    Call this at worker startup: configures tracing only if
    OTEL_EXPORTER_OTLP_ENDPOINT is set in the environment.
    """
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        service = os.environ.get("OTEL_SERVICE_NAME", "torchslicer-worker")
        configure(service_name=service)
