"""Verify ONNX execution providers through an isolated inference process."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Mapping, Sequence


PROBE_MODEL_SHA256 = "2ddb847dec08d0cc33422e1bb4b114c50a717ad38ca7bf3b930adea861296527"
_PROBE_MODEL_BASE64 = (
    "CAgSC252YnJvYWRjYXN0GgExOnYKIwoFaW5wdXQKBWlucHV0EgZvdXRwdXQaBnNx"
    "dWFyZSIDTXVsEhxudmJyb2FkY2FzdC1ydW50aW1lLXByb2JlLXYxWhcKBWlucHV0"
    "Eg4KDAgBEggKAggBCgIIBGIYCgZvdXRwdXQSDgoMCAESCAoCCAEKAggEQgQKABAN"
)
PROBE_INPUT = (1.0, 2.0, 3.0, 4.0)
PROBE_EXPECTED_OUTPUT = (1.0, 4.0, 9.0, 16.0)
PROBE_OUTPUT_SHAPE = (1, 4)
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
_RESULT_PREFIX = "NVBROADCAST_RUNTIME_PROBE_RESULT="
_MAX_DIAGNOSTIC_CHARS = 65_536
_CHILD_BOOTSTRAP = (
    "import runpy, sys; "
    "package_root = sys.argv.pop(1); "
    "probe_path = sys.argv.pop(1); "
    "sys.path.insert(0, package_root); "
    "runpy.run_path(probe_path, run_name='__main__')"
)


class ProbeProvider(StrEnum):
    """Execution providers whose readiness affects supported app modes."""

    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"

    @property
    def ort_name(self) -> str:
        return {
            ProbeProvider.CPU: "CPUExecutionProvider",
            ProbeProvider.CUDA: "CUDAExecutionProvider",
            ProbeProvider.TENSORRT: "TensorrtExecutionProvider",
        }[self]


@dataclass(frozen=True)
class RuntimeProbeResult:
    """Structured result returned by a fresh provider-probe process."""

    provider: ProbeProvider
    success: bool
    available_providers: tuple[str, ...] = ()
    registered_providers: tuple[str, ...] = ()
    executed_providers: tuple[str, ...] = ()
    output: tuple[float, ...] = ()
    output_shape: tuple[int, ...] = ()
    model_sha256: str = PROBE_MODEL_SHA256
    error: str = ""
    traceback: str = ""
    diagnostics: str = ""

    @classmethod
    def failure(
        cls,
        provider: ProbeProvider,
        error: str,
        *,
        diagnostics: str = "",
    ) -> RuntimeProbeResult:
        return cls(
            provider=provider,
            success=False,
            error=error,
            diagnostics=_bounded_diagnostic(diagnostics),
        )

    @property
    def failure_detail(self) -> str:
        """Return the complete bounded failure information for user surfaces."""
        details = []
        if self.error.strip():
            details.append(self.error.strip())
        if self.diagnostics.strip():
            details.append(self.diagnostics.strip())
        if self.traceback.strip():
            details.append(self.traceback.strip())
        return _bounded_diagnostic("\n\n".join(details))

    def to_payload(self) -> dict[str, object]:
        return {
            "provider": self.provider.value,
            "success": self.success,
            "available_providers": list(self.available_providers),
            "registered_providers": list(self.registered_providers),
            "executed_providers": list(self.executed_providers),
            "output": list(self.output),
            "output_shape": list(self.output_shape),
            "model_sha256": self.model_sha256,
            "error": self.error,
            "traceback": self.traceback,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> RuntimeProbeResult:
        return cls(
            provider=ProbeProvider(str(payload["provider"])),
            success=payload.get("success") is True,
            available_providers=_string_tuple(payload.get("available_providers")),
            registered_providers=_string_tuple(payload.get("registered_providers")),
            executed_providers=_string_tuple(payload.get("executed_providers")),
            output=_float_tuple(payload.get("output")),
            output_shape=_int_tuple(payload.get("output_shape")),
            model_sha256=str(payload.get("model_sha256", "")),
            error=str(payload.get("error", "")),
            traceback=str(payload.get("traceback", "")),
            diagnostics=str(payload.get("diagnostics", "")),
        )


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _float_tuple(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value))


def _int_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in _sequence(value))


def _bounded_diagnostic(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    if len(text) <= _MAX_DIAGNOSTIC_CHARS:
        return text
    omitted = len(text) - _MAX_DIAGNOSTIC_CHARS
    return f"{text[:_MAX_DIAGNOSTIC_CHARS]}\n...[{omitted} characters omitted]"


def probe_model_bytes() -> bytes:
    """Return the embedded model only when its pinned digest is intact."""
    model = base64.b64decode(_PROBE_MODEL_BASE64, validate=True)
    digest = hashlib.sha256(model).hexdigest()
    if digest != PROBE_MODEL_SHA256:
        raise RuntimeError(
            "embedded runtime probe model checksum mismatch: "
            f"expected {PROBE_MODEL_SHA256}, found {digest}"
        )
    return model


def _provider_request(
    provider: ProbeProvider, device_id: int
) -> list[str | tuple[str, dict[str, object]]]:
    if provider is ProbeProvider.CPU:
        return [provider.ort_name]
    options: dict[str, object] = {"device_id": device_id}
    if provider is ProbeProvider.TENSORRT:
        options["trt_engine_cache_enable"] = False
    return [(provider.ort_name, options)]


def _profile_execution_providers(events: object) -> tuple[str, ...]:
    if not isinstance(events, list):
        return ()
    providers = {
        str(event.get("args", {}).get("provider"))
        for event in events
        if isinstance(event, dict)
        and isinstance(event.get("args"), dict)
        and event["args"].get("provider")
    }
    return tuple(sorted(providers))


def probe_observation_problems(
    provider: ProbeProvider,
    registered_providers: Sequence[str],
    executed_providers: Sequence[str],
    output: Sequence[float],
    output_shape: Sequence[int],
) -> list[str]:
    """Return failures proving the requested provider did not execute correctly."""
    problems: list[str] = []
    if provider.ort_name not in registered_providers:
        problems.append(
            f"requested provider {provider.ort_name} was not registered; "
            f"session registered {', '.join(registered_providers) or 'none'}"
        )
    if set(executed_providers) != {provider.ort_name}:
        problems.append(
            f"probe graph did not execute exclusively on {provider.ort_name}; "
            f"profile reported {', '.join(executed_providers) or 'none'}"
        )
    if tuple(output_shape) != PROBE_OUTPUT_SHAPE:
        problems.append(
            f"probe output shape was {tuple(output_shape)}, expected "
            f"{PROBE_OUTPUT_SHAPE}"
        )
    if len(output) != len(PROBE_EXPECTED_OUTPUT) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
        for actual, expected in zip(output, PROBE_EXPECTED_OUTPUT)
    ):
        problems.append(
            f"probe output was {tuple(output)}, expected {PROBE_EXPECTED_OUTPUT}"
        )
    return problems


def _preload_provider_libraries(provider: ProbeProvider) -> list[str]:
    if provider is ProbeProvider.CPU:
        return []

    from nvbroadcast.core.platform import preload_nvidia_runtime_libs

    preload_nvidia_runtime_libs()
    if provider is not ProbeProvider.TENSORRT:
        return []

    from nvbroadcast.core.platform import preload_tensorrt_runtime_libs

    return preload_tensorrt_runtime_libs()


def _root_exception(error: Exception) -> Exception:
    current = error
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_error = current.__cause__ or current.__context__
        if not isinstance(next_error, Exception):
            break
        current = next_error
    return current


def execute_provider_probe(
    provider: ProbeProvider, device_id: int = 0
) -> RuntimeProbeResult:
    """Execute one provider probe in the current, already-isolated process."""
    if device_id < 0:
        return RuntimeProbeResult.failure(
            provider, f"GPU device index must be non-negative, found {device_id}"
        )

    available_providers: tuple[str, ...] = ()
    registered_providers: tuple[str, ...] = ()
    executed_providers: tuple[str, ...] = ()
    output: tuple[float, ...] = ()
    output_shape: tuple[int, ...] = ()
    preload_errors: list[str] = []
    session = None
    profile_ended = False

    try:
        model = probe_model_bytes()
        preload_errors = _preload_provider_libraries(provider)

        import numpy as np
        import onnxruntime as ort

        available_providers = tuple(ort.get_available_providers())
        if provider.ort_name not in available_providers:
            return RuntimeProbeResult(
                provider=provider,
                success=False,
                available_providers=available_providers,
                error=(
                    f"{provider.ort_name} is not available; ONNX Runtime exposes "
                    f"{', '.join(available_providers) or 'no providers'}"
                ),
                diagnostics="\n".join(preload_errors),
            )

        with tempfile.TemporaryDirectory(prefix="nvbroadcast-runtime-probe-") as tmp:
            options = ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(tmp) / "profile")
            if provider is not ProbeProvider.CPU:
                options.add_session_config_entry(
                    "session.disable_cpu_ep_fallback", "1"
                )

            session = ort.InferenceSession(
                model,
                sess_options=options,
                providers=_provider_request(provider, device_id),
            )
            session.disable_fallback()
            registered_providers = tuple(session.get_providers())
            input_array = np.asarray([PROBE_INPUT], dtype=np.float32)
            output_array = session.run(
                ["output"], {"input": input_array}
            )[0]
            output_shape = tuple(int(dimension) for dimension in output_array.shape)
            output = tuple(float(item) for item in output_array.reshape(-1))

            profile_path = Path(session.end_profiling())
            profile_ended = True
            events = json.loads(profile_path.read_text(encoding="utf-8"))
            executed_providers = _profile_execution_providers(events)

        problems = probe_observation_problems(
            provider,
            registered_providers,
            executed_providers,
            output,
            output_shape,
        )
        return RuntimeProbeResult(
            provider=provider,
            success=not problems,
            available_providers=available_providers,
            registered_providers=registered_providers,
            executed_providers=executed_providers,
            output=output,
            output_shape=output_shape,
            error="; ".join(problems),
            diagnostics="\n".join(preload_errors),
        )
    except Exception as error:
        root_error = _root_exception(error)
        return RuntimeProbeResult(
            provider=provider,
            success=False,
            available_providers=available_providers,
            registered_providers=registered_providers,
            executed_providers=executed_providers,
            output=output,
            output_shape=output_shape,
            error=f"{type(root_error).__name__}: {root_error}",
            traceback=traceback.format_exc(),
            diagnostics="\n".join(preload_errors),
        )
    finally:
        if session is not None and not profile_ended:
            try:
                session.end_profiling()
            except Exception:
                pass


def _parse_child_result(
    provider: ProbeProvider, completed: subprocess.CompletedProcess[str]
) -> RuntimeProbeResult:
    if isinstance(completed.stdout, bytes):
        raw_stdout = completed.stdout.decode("utf-8", errors="replace")
    else:
        raw_stdout = str(completed.stdout or "")
    stderr = _bounded_diagnostic(completed.stderr)
    payload_line = next(
        (
            line
            for line in reversed(raw_stdout.splitlines())
            if line.startswith(_RESULT_PREFIX)
        ),
        "",
    )
    child_stdout = _bounded_diagnostic(
        "\n".join(
            line
            for line in raw_stdout.splitlines()
            if not line.startswith(_RESULT_PREFIX)
        ).strip()
    )
    process_diagnostics = "\n".join(
        part
        for part in (
            f"probe stdout:\n{child_stdout}" if child_stdout else "",
            f"probe stderr:\n{stderr.strip()}" if stderr.strip() else "",
        )
        if part
    )

    if not payload_line:
        return RuntimeProbeResult.failure(
            provider,
            f"probe process exited with code {completed.returncode} without a result",
            diagnostics=process_diagnostics,
        )
    try:
        payload = json.loads(payload_line.removeprefix(_RESULT_PREFIX))
        if not isinstance(payload, dict):
            raise TypeError("probe payload is not an object")
        result = RuntimeProbeResult.from_payload(payload)
    except Exception as error:
        return RuntimeProbeResult.failure(
            provider,
            f"invalid probe result: {type(error).__name__}: {error}",
            diagnostics=process_diagnostics,
        )

    diagnostics = "\n".join(
        part for part in (result.diagnostics.strip(), process_diagnostics) if part
    )
    result = replace(result, diagnostics=_bounded_diagnostic(diagnostics))
    if result.provider is not provider:
        return replace(
            result,
            success=False,
            error=(
                f"probe returned provider {result.provider.value}, expected "
                f"{provider.value}"
            ),
        )
    if result.model_sha256 != PROBE_MODEL_SHA256:
        return replace(
            result,
            success=False,
            error=(
                f"probe returned model digest {result.model_sha256 or 'none'}, "
                f"expected {PROBE_MODEL_SHA256}"
            ),
        )
    if result.success:
        problems = probe_observation_problems(
            provider,
            result.registered_providers,
            result.executed_providers,
            result.output,
            result.output_shape,
        )
        if problems:
            return replace(
                result,
                success=False,
                error="invalid successful probe result: " + "; ".join(problems),
            )
    if completed.returncode != 0 and result.success:
        return replace(
            result,
            success=False,
            error=f"probe reported success but exited with code {completed.returncode}",
        )
    return result


def _run_provider_probe(
    provider: ProbeProvider,
    device_id: int,
    timeout: float,
    executable: str,
) -> RuntimeProbeResult:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    probe_path = Path(__file__).resolve()
    package_root = probe_path.parents[2]
    command = [
        executable,
        "-I",
        "-c",
        _CHILD_BOOTSTRAP,
        str(package_root),
        str(probe_path),
        "--provider",
        provider.value,
        "--device-id",
        str(device_id),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        diagnostics = "\n".join(
            part
            for part in (
                f"probe stdout:\n{_bounded_diagnostic(error.stdout)}"
                if error.stdout
                else "",
                f"probe stderr:\n{_bounded_diagnostic(error.stderr)}"
                if error.stderr
                else "",
            )
            if part
        )
        return RuntimeProbeResult.failure(
            provider,
            f"provider execution probe timed out after {timeout:g} seconds",
            diagnostics=diagnostics,
        )
    except Exception as error:
        return RuntimeProbeResult.failure(
            provider,
            f"could not start provider execution probe: {type(error).__name__}: {error}",
        )
    return _parse_child_result(provider, completed)


@lru_cache(maxsize=None)
def _cached_provider_probe(
    provider: ProbeProvider,
    device_id: int,
    timeout: float,
    executable: str,
) -> RuntimeProbeResult:
    return _run_provider_probe(provider, device_id, timeout, executable)


def probe_execution_provider(
    provider: ProbeProvider,
    device_id: int = 0,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    *,
    use_cache: bool = True,
) -> RuntimeProbeResult:
    """Probe a provider in a fresh interpreter and return structured evidence."""
    if device_id < 0:
        return RuntimeProbeResult.failure(
            provider, f"GPU device index must be non-negative, found {device_id}"
        )
    if timeout <= 0:
        return RuntimeProbeResult.failure(
            provider, f"probe timeout must be positive, found {timeout:g}"
        )
    arguments = (provider, device_id, float(timeout), sys.executable)
    if use_cache:
        return _cached_provider_probe(*arguments)
    return _run_provider_probe(*arguments)


def clear_runtime_probe_cache() -> None:
    """Invalidate process-local results after an environment mutation."""
    _cached_provider_probe.cache_clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=tuple(ProbeProvider))
    parser.add_argument("--device-id", type=int, default=0)
    arguments = parser.parse_args()
    result = execute_provider_probe(
        ProbeProvider(arguments.provider), arguments.device_id
    )
    print(_RESULT_PREFIX + json.dumps(result.to_payload(), sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
