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

import atexit
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
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
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
        # W3C TraceContext propagator — required for inject()/extract() to write/read
        # the 'traceparent' header in gRPC metadata.  Without this, inject() is a no-op
        # and every span appears as an orphaned root in Phoenix.
        set_global_textmap(CompositePropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator(),
        ]))
        # Flush all pending spans on process exit so short-lived processes
        # (e.g. coordinator with keep_alive=false) don't drop spans.
        atexit.register(provider.shutdown)
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
def span(name: str, *, kind: str | None = None, new_trace: bool = False, **attrs):
    """
    Context manager for a traced span.

    Parameters
    ----------
    name:
        Span name.
    kind:
        Optional OpenInference span kind (e.g. ``"CHAIN"``, ``"TOOL"``).
        Sets the ``openinference.span.kind`` attribute, which controls how
        Phoenix labels the span in its chain view.
    new_trace:
        If True, start this span as a new root trace (ignoring any current
        span context).  Use this for logical units that should appear as
        separate traces in Phoenix — e.g. one trace per training batch.
    **attrs:
        Additional span attributes (coerced to OTEL-supported types).

    Usage::

        with tracer.span("worker.forward", kind="TOOL", batch_id=42) as s:
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

    # new_trace=True → detached context so this span starts a fresh trace root
    try:
        from opentelemetry import context as _otel_ctx
        _ctx = _otel_ctx.Context() if new_trace else None
    except Exception:
        _ctx = None

    try:
        with _tracer.start_as_current_span(name, context=_ctx) as s:
            # Set OpenInference span kind (for Phoenix chain view)
            if kind is not None:
                _safe_set(s, "openinference.span.kind", kind.upper())
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


def inject_context_bytes() -> bytes:
    """Serialize the current span context to a W3C traceparent ASCII string.

    Used to propagate trace context over transports that don't support gRPC
    metadata (e.g. the raw TCP tensor channel).  Returns ``b''`` immediately
    when tracing is disabled — the caller pays no overhead beyond the check.
    """
    if _tracer is None:
        return b""
    try:
        from opentelemetry import propagate as _prop
        carrier: dict = {}
        _prop.inject(carrier)
        tp = carrier.get("traceparent", "")
        return tp.encode("ascii") if tp else b""
    except Exception:
        return b""


class _ExtractContextCM:
    """Attach a propagated trace context for the lifetime of a ``with`` block.

    Instantiated by :func:`extract_context`.  When tracing is disabled or
    ``ctx_bytes`` is empty this is a zero-overhead no-op object.
    """
    __slots__ = ("_ctx_bytes", "_token")

    def __init__(self, ctx_bytes: bytes):
        self._ctx_bytes = ctx_bytes
        self._token = None

    def __enter__(self):
        if not self._ctx_bytes or _tracer is None:
            return self
        try:
            from opentelemetry import propagate as _prop, context as _ctx
            carrier = {"traceparent": self._ctx_bytes.decode("ascii")}
            self._token = _ctx.attach(_prop.extract(carrier))
        except Exception:
            pass
        return self

    def __exit__(self, *_):
        if self._token is not None:
            try:
                from opentelemetry import context as _ctx
                _ctx.detach(self._token)
            except Exception:
                pass
        return False


def extract_context(ctx_bytes: bytes) -> _ExtractContextCM:
    """Return a context manager that restores a propagated trace context.

    Designed for the TCP tensor transport: the receiver calls this with the
    bytes returned by the sender's :func:`inject_context_bytes` call, making
    any spans created inside the block children of the original coordinator
    batch span.  Zero-overhead no-op when tracing is disabled.
    """
    return _ExtractContextCM(ctx_bytes)


def propagate_to_thread(fn):
    """Wrap *fn* so it inherits the current OTEL context when run in a new thread.

    Python's ``threading.Thread`` does not propagate ``contextvars``, so
    gRPC callbacks fired from worker threads lose their parent span.  Wrap the
    target with this before passing it to ``threading.Thread(target=...)``.
    Returns *fn* unchanged when tracing is disabled.
    """
    if _tracer is None:
        return fn
    try:
        from opentelemetry import context as _ctx
        _current = _ctx.get_current()
        def _wrapped(*args, **kwargs):
            token = _ctx.attach(_current)
            try:
                return fn(*args, **kwargs)
            finally:
                _ctx.detach(token)
        return _wrapped
    except Exception:
        return fn


def auto_configure_if_env() -> None:
    """
    Call this at worker startup: configures tracing only if
    OTEL_EXPORTER_OTLP_ENDPOINT is set in the environment.
    """
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        service = os.environ.get("OTEL_SERVICE_NAME", "torchslicer-worker")
        configure(service_name=service)
