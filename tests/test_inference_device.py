"""RVM stream/device ownership without CUDA, model or device execution."""
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

from nvbroadcast.video.effects import VideoEffects


class _CudaError(RuntimeError):
    pass


class _Devices:
    def __init__(self):
        self.local = threading.local()
        self.selections, self.failures = [], []
        self.stack = []
        self.cuda = SimpleNamespace(
            Device=self.device,
            runtime=SimpleNamespace(getDevice=lambda: self.current, setDevice=self.set_device),
        )

    @property
    def current(self):
        return getattr(self.local, "current", 0)

    @current.setter
    def current(self, value):
        self.local.current = value

    def set_device(self, selected):
        self.selections.append(selected)
        if selected == 1 and self.failures:
            raise self.failures.pop(0)
        self.current = selected

    def device(self, selected):
        owner = self

        class Device:
            def __enter__(self):
                # CuPy 14.2 pushes its TLS stack before selection can fail.
                owner.stack.append(owner.current)
                owner.set_device(selected)
                return self

            def __exit__(self, kind, error, traceback):
                owner.set_device(owner.stack.pop())

        return Device()


class InferenceDeviceTests(unittest.TestCase):
    def make_effects(self, providers=("CUDAExecutionProvider",), *, cupy=True):
        effects = VideoEffects.__new__(VideoEffects)
        cp = _Devices()
        effects._cupy = cp if cupy else None
        effects._gpu_index = 1
        alpha, frame, history = object(), object(), object()
        observed = []
        session = SimpleNamespace(get_providers=lambda: list(providers))

        def infer(value, width, height):
            observed.append(cp.current)
            self.assertIs(value, frame)
            self.assertEqual((width, height), (1280, 720))
            if providers and providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                if cp.current != 1:
                    raise _CudaError("CUDA error invalid resource handle on reused stream")
            return alpha

        backend = SimpleNamespace(
            session=session, infer=mock.Mock(side_effect=infer), history=history,
            _is_cuda_runtime_error=lambda error: isinstance(error, _CudaError),
            _recover_cuda_session=mock.Mock(return_value=False),
        )
        effects._backend = backend
        effects._lock = threading.RLock()
        effects._engine_reload_in_progress = False
        effects._gpu_matte_eligible = lambda: False
        effects._bg_mode = "blur"
        effects._refine_alpha = lambda value: value
        effects._edge_refine_enabled = False
        effects._temporal_smooth = lambda value, version: value
        return effects, cp, backend, frame, alpha, history, observed

    def test_fresh_callback_binds_reused_gpu_session_and_restores_caller(self):
        effects, cp, backend, frame, alpha, history, observed = self.make_effects()
        cp.current = 3
        values, errors = [], []

        def callback():
            try:
                values.append(cp.current)
                values.append(effects._run_inference(frame, 1280, 720, 17))
                values.append(cp.current)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=callback)
        worker.start()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(values[0], 0)
        self.assertIs(values[1], alpha)
        self.assertEqual(values[2], 0)
        self.assertEqual(cp.current, 3)
        self.assertEqual(observed, [1])
        self.assertIs(backend.history, history)
        backend._recover_cuda_session.assert_not_called()

    def test_internal_inference_retries_keep_selected_device(self):
        effects, cp, backend, frame, alpha, _history, observed = self.make_effects()

        def retrying(value, width, height):
            self.assertIs(value, frame)
            for _ in range(3):
                observed.append(cp.current)
                self.assertEqual(cp.current, 1)
            return alpha

        backend.infer.side_effect = retrying
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(observed, [1, 1, 1])
        self.assertEqual(cp.current, 0)
        self.assertEqual(cp.selections, [1, 0])

    def test_tensorrt_primary_session_is_bound(self):
        effects, cp, backend, frame, alpha, _history, observed = self.make_effects(("TensorrtExecutionProvider", "CUDAExecutionProvider"))
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(observed, [1])
        self.assertEqual(cp.current, 0)

    def test_body_exception_restores_device_without_duplicate_recovery(self):
        effects, cp, backend, frame, _alpha, _history, _observed = self.make_effects()
        error = _CudaError("generated inference failure after its own recovery")
        backend.infer.side_effect = error
        with self.assertRaises(_CudaError) as raised:
            effects._infer_numpy_backend(backend, frame, 1280, 720)
        self.assertIs(raised.exception, error)
        self.assertEqual(cp.current, 0)
        self.assertEqual(cp.selections, [1, 0])
        backend._recover_cuda_session.assert_not_called()

    def test_cpu_coreml_and_missing_provider_skip_device_selection(self):
        for providers in (("CPUExecutionProvider",), ("CoreMLExecutionProvider", "CPUExecutionProvider"), ("CPUExecutionProvider", "CUDAExecutionProvider"), ()):
            with self.subTest(providers=providers):
                effects, cp, backend, frame, alpha, history, observed = self.make_effects(providers)
                self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
                self.assertEqual(cp.selections, [])
                self.assertEqual(observed, [0])
                self.assertIs(backend.history, history)

    def test_no_cupy_preserves_existing_backend_call(self):
        effects, cp, backend, frame, alpha, _history, _observed = self.make_effects(cupy=False)
        backend.infer.side_effect = None
        backend.infer.return_value = alpha
        backend.session.get_providers = mock.Mock(side_effect=AssertionError("No capability query needed"))
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        backend.infer.assert_called_once_with(frame, 1280, 720)
        backend.session.get_providers.assert_not_called()
        self.assertEqual(cp.selections, [])

    def test_non_ort_backend_preserves_existing_call(self):
        effects, cp, _backend, frame, alpha, _history, _observed = self.make_effects()
        backend = SimpleNamespace(infer=mock.Mock(return_value=alpha))
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        backend.infer.assert_called_once_with(frame, 1280, 720)
        self.assertEqual(cp.selections, [])

    def test_gpu_backend_without_rvm_recovery_contract_is_unchanged(self):
        effects, cp, _backend, frame, alpha, _history, _observed = self.make_effects()
        backend = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"]), infer=mock.Mock(return_value=alpha))
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(cp.selections, [])

    def test_binding_failure_uses_existing_recovery_then_cpu_fallback(self):
        effects, cp, backend, frame, alpha, _history, _observed = self.make_effects()
        error = _CudaError("CUDA error device selection failed")
        cp.failures = [error]

        def recover(value):
            self.assertIs(value, error)
            cp.current = 1  # ORT session construction can select the GPU.
            backend.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
            backend.infer.side_effect = None
            backend.infer.return_value = alpha
            return True

        backend._recover_cuda_session.side_effect = recover
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(cp.selections, [1, 0])
        self.assertEqual(cp.stack, [])
        backend._recover_cuda_session.assert_called_once_with(error)
        backend.infer.assert_called_once_with(frame, 1280, 720)
        self.assertEqual(cp.current, 0)

    def test_binding_failure_retries_recreated_gpu_context_once(self):
        effects, cp, backend, frame, alpha, _history, observed = self.make_effects()
        error = _CudaError("CUDA error first binding failed")
        cp.failures = [error]
        def recover(value):
            cp.current = 1
            return True

        backend._recover_cuda_session.side_effect = recover
        self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(cp.selections, [1, 1, 0])
        self.assertEqual(cp.stack, [])
        self.assertEqual(observed, [1])
        backend._recover_cuda_session.assert_called_once_with(error)
        self.assertEqual(cp.current, 0)

    def test_second_binding_failure_does_not_loop_or_infer_unguarded(self):
        effects, cp, backend, frame, _alpha, _history, _observed = self.make_effects()
        first, second = _CudaError("CUDA error first"), _CudaError("CUDA error second")
        cp.failures = [first, second]
        backend._recover_cuda_session.return_value = True
        with self.assertRaises(_CudaError) as raised:
            effects._infer_numpy_backend(backend, frame, 1280, 720)
        self.assertIs(raised.exception, second)
        self.assertEqual(cp.selections, [1, 1, 0])
        self.assertEqual(cp.current, 0)
        self.assertEqual(cp.stack, [])
        backend._recover_cuda_session.assert_called_once_with(first)
        backend.infer.assert_not_called()

    def test_failed_or_throwing_recovery_never_calls_backend_unguarded(self):
        for throws in (False, True):
            with self.subTest(throws=throws):
                effects, cp, backend, frame, _alpha, _history, _observed = self.make_effects()
                cp.failures = [_CudaError("CUDA error entry failed")]
                if throws:
                    backend._recover_cuda_session.side_effect = RuntimeError("generated recovery failure")
                with self.assertRaises(RuntimeError):
                    effects._infer_numpy_backend(backend, frame, 1280, 720)
                backend._recover_cuda_session.assert_called_once()
                backend.infer.assert_not_called()
                self.assertEqual(cp.selections, [1, 0])
                self.assertEqual(cp.current, 0)
                self.assertEqual(cp.stack, [])

    def test_non_cuda_entry_failure_is_not_treated_as_runtime_recovery(self):
        effects, cp, backend, frame, _alpha, _history, _observed = self.make_effects()
        cp.failures = [ValueError("generated programming error")]
        with self.assertRaises(ValueError):
            effects._infer_numpy_backend(backend, frame, 1280, 720)
        backend._recover_cuda_session.assert_not_called()
        backend.infer.assert_not_called()
        self.assertEqual(cp.current, 0)
        self.assertEqual(cp.stack, [])

    def test_normal_inference_inside_exception_handler_restores_device(self):
        effects, cp, backend, frame, alpha, _history, _observed = self.make_effects()
        try:
            raise ValueError("unrelated caller exception")
        except ValueError:
            self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
        self.assertEqual(cp.current, 0)
        self.assertEqual(cp.selections, [1, 0])
        self.assertEqual(cp.stack, [])

    def test_cuda_restore_failure_preserves_cpu_result_and_body_exception(self):
        for body_error in (None, ValueError("generated CPU inference failure")):
            with self.subTest(body_error=body_error):
                effects, cp, backend, frame, alpha, _history, _observed = self.make_effects()
                cp.failures = [_CudaError("CUDA error selection failed")]
                set_device = cp.cuda.runtime.setDevice

                def select(value):
                    if value == 0:
                        raise _CudaError("CUDA error driver unavailable during restore")
                    set_device(value)

                def recover(error):
                    backend.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
                    backend.infer.side_effect = body_error
                    backend.infer.return_value = alpha
                    return True

                cp.cuda.runtime.setDevice = select
                backend._recover_cuda_session.side_effect = recover
                if body_error is None:
                    self.assertIs(effects._infer_numpy_backend(backend, frame, 1280, 720), alpha)
                else:
                    with self.assertRaises(ValueError) as raised:
                        effects._infer_numpy_backend(backend, frame, 1280, 720)
                    self.assertIs(raised.exception, body_error)
                backend._recover_cuda_session.assert_called_once()
                backend.infer.assert_called_once_with(frame, 1280, 720)
                self.assertEqual(cp.stack, [])

    def test_restore_errors_are_not_hidden_for_gpu_or_non_cuda_failures(self):
        for providers, error in ((["CUDAExecutionProvider"], _CudaError("CUDA error restore failed")), (["CPUExecutionProvider"], ValueError("generated restoration error"))):
            with self.subTest(providers=providers):
                effects, cp, backend, frame, alpha, _history, _observed = self.make_effects()
                set_device = cp.cuda.runtime.setDevice

                def select(value):
                    if value == 0:
                        raise error
                    set_device(value)

                def infer(*args):
                    backend.session = SimpleNamespace(get_providers=lambda: providers)
                    return alpha

                cp.cuda.runtime.setDevice = select
                backend.infer.side_effect = infer
                with self.assertRaises(type(error)) as raised:
                    effects._infer_numpy_backend(backend, frame, 1280, 720)
                self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
