"""Own disposable provider-probe children for one application's lifetime."""

from contextlib import contextmanager
import os
import signal
import subprocess
import threading
import time


_ACTIVE_SCOPE = threading.local()


class ProviderProbeCancelled(RuntimeError):
    """Owner cancellation is not a failed provider and must not be cached."""


def _kill_owned_group(child: subprocess.Popen) -> None:
    """Only accept a child created in a private session by this module."""
    try:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
    except ProcessLookupError:
        pass


class ProviderProbeScope:
    """Cancel/reap this owner's probes without touching unrelated processes."""

    def __init__(self, parent=None):
        self._lock = parent._lock if parent is not None else threading.Lock()
        self._closed = False
        self._children: set[subprocess.Popen] = set()
        self._scopes: set[ProviderProbeScope] = set()
        if parent is not None:
            with self._lock:
                if parent._closed:
                    raise ProviderProbeCancelled("Provider probe owner is closed")
                parent._scopes.add(self)

    def child(self):
        """Keep one dialog's cancellation inside its application owner."""
        return ProviderProbeScope(parent=self)

    @contextmanager
    def activate(self):
        with self._lock:
            if self._closed:
                raise ProviderProbeCancelled("Provider probe owner is closed")
        previous = getattr(_ACTIVE_SCOPE, "owner", None)
        _ACTIVE_SCOPE.owner = self
        try:
            yield
        finally:
            _ACTIVE_SCOPE.owner = previous

    def run(self, command, *, env, stdin, timeout, check, capture_output, text):
        # This implements the fixed capture contract used by runtime.probe;
        # it is not a replacement for arbitrary application subprocesses.
        if check or not capture_output or not text:
            raise ValueError("Unexpected provider-probe capture options")
        with self._lock:
            if self._closed:
                raise ProviderProbeCancelled("Provider probe owner is closed")
            # Registration and close's state transition are atomic. Never
            # hold this lock while communicating with or waiting for a child.
            child = subprocess.Popen(
                command, env=env, stdin=stdin,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=os.name == "posix",
            )
            self._children.add(child)
        try:
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_owned_group(child)
                stdout, stderr = child.communicate(timeout=2.0)
                with self._lock:
                    if self._closed:
                        raise ProviderProbeCancelled("Provider probe owner is closed")
                raise subprocess.TimeoutExpired(command, timeout, stdout, stderr)
            with self._lock:
                if self._closed:
                    raise ProviderProbeCancelled("Provider probe owner is closed")
            return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
        finally:
            # A direct child may have exited while a descendant still owns
            # its session or inherited pipes. Dispose of the whole owned group.
            _kill_owned_group(child)
            try:
                child.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            else:
                with self._lock:
                    self._children.discard(child)
            for pipe in (child.stdout, child.stderr):
                if pipe is not None:
                    pipe.close()

    def close(self, timeout: float = 2.0) -> bool:
        """Signal every owned group, then reap within one shared deadline."""
        deadline = time.monotonic() + timeout
        with self._lock:
            scopes = [self]
            children = set()
            while scopes:
                scope = scopes.pop()
                scope._closed = True
                children.update(scope._children)
                scopes.extend(scope._scopes)
            children = tuple(children)
        for child in children:
            _kill_owned_group(child)
        stopped = True
        for child in children:
            try:
                child.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                stopped = False
        return stopped


def check_probe_cancellation() -> None:
    """Keep a cancelled transport result out of the provider-result cache."""
    owner = getattr(_ACTIVE_SCOPE, "owner", None)
    if owner is not None:
        with owner._lock:
            if owner._closed:
                raise ProviderProbeCancelled("Provider probe owner is closed")


def run_probe_process(*args, **kwargs):
    """Keep default probes unchanged; only active owners use managed children."""
    owner = getattr(_ACTIVE_SCOPE, "owner", None)
    if owner is None:
        return subprocess.run(*args, **kwargs)
    return owner.run(*args, **kwargs)
