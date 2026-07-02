import unittest
import numpy as np
import paddle
import paddle.nn.functional as F


def log_softmax_numpy_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x64 = x.astype(np.float64)
    x_max = np.max(x64, axis=axis, keepdims=True)
    y = x64 - x_max
    lse = np.log(np.sum(np.exp(y), axis=axis, keepdims=True)) + x_max
    out = x64 - lse
    return out.astype(x.dtype, copy=False)


def assert_allclose(a, b, rtol=1e-5, atol=1e-6, msg=""):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def logsumexp_np(a: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    a64 = a.astype(np.float64)
    m = np.max(a64, axis=axis, keepdims=True)
    s = np.sum(np.exp(a64 - m), axis=axis, keepdims=True)
    out = np.log(s) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class TestLogSoftmaxUT(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        np.random.seed(2026)

    def _property_checks(self, y_np: np.ndarray, axis: int, dtype: str, shape):
        self.assertTrue(np.isfinite(y_np).all(), "found NaN/Inf in output")

        lse = logsumexp_np(y_np, axis=axis, keepdims=False)
        if dtype == "float16":
            assert_allclose(
                lse, np.zeros_like(lse),
                rtol=0.0, atol=8e-4,
                msg=f"logsumexp != 0 (fp16): shape={shape} axis={axis}"
            )
        else:
            assert_allclose(
                lse, np.zeros_like(lse),
                rtol=0.0, atol=1e-6,
                msg=f"logsumexp != 0: shape={shape} axis={axis}"
            )

        expy = np.exp(y_np.astype(np.float64))
        s = np.sum(expy, axis=axis, keepdims=False)
        if dtype == "float16":
            assert_allclose(
                s, np.ones_like(s),
                rtol=0.0, atol=8e-4,
                msg=f"prob sum != 1 (fp16): shape={shape} axis={axis}"
            )
        else:
            assert_allclose(
                s, np.ones_like(s),
                rtol=1e-6, atol=1e-6,
                msg=f"prob sum != 1: shape={shape} axis={axis}"
            )

    def _run_case(self, shape, axis, dtype="float32", rtol=1e-5, atol=1e-6):
        x_np = (np.random.randn(*shape) * 3.0).astype(dtype)
        x_np = x_np + (np.random.uniform(-50, 50, size=shape).astype(dtype) * 0.02)

        x = paddle.to_tensor(x_np, stop_gradient=False)

        y = F.log_softmax(x, axis=axis)
        y_np = y.numpy()

        ref = log_softmax_numpy_stable(x_np, axis=axis)

        assert_allclose(
            y_np, ref, rtol=rtol, atol=atol,
            msg=f"mismatch: shape={shape} axis={axis} dtype={dtype}"
        )

        self._property_checks(y_np, axis=axis, dtype=dtype, shape=shape)

    def test_forward_float32_multi_axes(self):
        self._run_case(shape=(2, 3, 5), axis=-1, dtype="float32", rtol=1e-6, atol=1e-6)
        self._run_case(shape=(2, 3, 5), axis=0, dtype="float32", rtol=1e-6, atol=1e-6)
        self._run_case(shape=(4, 7), axis=1, dtype="float32", rtol=1e-6, atol=1e-6)

    def test_forward_float16(self):
        self._run_case(shape=(2, 8, 16), axis=-1, dtype="float16", rtol=5e-3, atol=5e-3)

    def test_extreme_values(self):
        x_np = np.array([
            [1000.0, 0.0, -1000.0],
            [88.0, 88.0, 88.0],
            [-1000.0, -1001.0, -999.0],
        ], dtype=np.float32)
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.log_softmax(x, axis=-1).numpy()
        ref = log_softmax_numpy_stable(x_np, axis=-1)

        assert_allclose(y, ref, rtol=1e-6, atol=1e-6, msg="extreme mismatch")
        self._property_checks(y, axis=-1, dtype="float32", shape=x_np.shape)

    def test_backward_gradcheck_float32(self):
        shape = (2, 3, 4)
        axis = -1
        eps = 1e-3

        rtol = 2e-2
        atol = 3e-3

        x_np = (np.random.randn(*shape) * 1.7).astype(np.float32)
        w_np = np.random.randn(*shape).astype(np.float32)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        w = paddle.to_tensor(w_np)

        y = F.log_softmax(x, axis=axis)
        loss = paddle.sum(y * w)
        loss.backward()
        grad_pd = x.grad.numpy().astype(np.float64)

        sum_axis = np.sum(grad_pd, axis=axis)
        np.testing.assert_allclose(
            sum_axis, np.zeros_like(sum_axis),
            rtol=0.0, atol=1e-6,
            err_msg="analytic property failed: sum(grad, axis) != 0"
        )

        rng = np.random.RandomState(2026)
        num_checks = 12
        flat_size = x_np.size
        idxs = rng.choice(flat_size, size=min(num_checks, flat_size), replace=False)

        def f(x_arr: np.ndarray) -> float:
            y_arr = log_softmax_numpy_stable(x_arr, axis=axis).astype(np.float64)
            return float(np.sum(y_arr * w_np.astype(np.float64)))

        grad_num = np.zeros_like(grad_pd)

        for flat_i in idxs:
            idx = np.unravel_index(int(flat_i), x_np.shape)
            x_pos = x_np.copy()
            x_neg = x_np.copy()
            x_pos[idx] += eps
            x_neg[idx] -= eps

            f_pos = f(x_pos)
            f_neg = f(x_neg)
            grad_num[idx] = (f_pos - f_neg) / (2.0 * eps)

        for flat_i in idxs:
            idx = np.unravel_index(int(flat_i), x_np.shape)
            a = grad_pd[idx]
            b = grad_num[idx]
            if not np.isfinite(a) or not np.isfinite(b):
                raise AssertionError(f"non-finite grad at idx={idx}: pd={a} num={b}")
            if not np.isclose(a, b, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"gradcheck failed at idx={idx}: pd={a} num={b} "
                    f"(rtol={rtol}, atol={atol})"
                )

    def test_mobile_net_v3_class_dim_1000(self):
        self._run_case(shape=(8, 1000), axis=-1, dtype="float32", rtol=1e-6, atol=1e-6)

    def test_backward_mobile_net_v3_class_dim_1000(self):
        shape = (4, 1000)
        axis = -1
        x_np = (np.random.randn(*shape) * 1.7).astype(np.float32)
        w_np = np.random.randn(*shape).astype(np.float32)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        w = paddle.to_tensor(w_np)

        y = F.log_softmax(x, axis=axis)
        loss = paddle.sum(y * w)
        loss.backward()
        grad_pd = x.grad.numpy().astype(np.float64)

        sum_axis = np.sum(grad_pd, axis=axis)
        np.testing.assert_allclose(
            sum_axis, np.zeros_like(sum_axis),
            rtol=0.0, atol=2e-5,
            err_msg="analytic property failed: sum(grad, axis) != 0"
        )

    def test_ppocrv5_server_rec_dim_6629_cpu_compare(self):
        shape = (2, 6629)
        axis = 1
        x_np = (np.random.randn(*shape) * 1.7).astype(np.float32)
        w_np = np.random.randn(*shape).astype(np.float32)

        paddle.set_device("cpu")
        x_cpu = paddle.to_tensor(x_np, stop_gradient=False)
        w_cpu = paddle.to_tensor(w_np)
        y_cpu = F.log_softmax(x_cpu, axis=axis)
        loss_cpu = paddle.sum(y_cpu * w_cpu)
        loss_cpu.backward()
        y_cpu_np = y_cpu.numpy()
        grad_cpu_np = x_cpu.grad.numpy()

        paddle.set_device("musa")
        x_musa = paddle.to_tensor(x_np, stop_gradient=False)
        w_musa = paddle.to_tensor(w_np)
        y_musa = F.log_softmax(x_musa, axis=axis)
        loss_musa = paddle.sum(y_musa * w_musa)
        loss_musa.backward()
        y_musa_np = y_musa.numpy()
        grad_musa_np = x_musa.grad.numpy()

        self.assertTrue(np.isfinite(y_musa_np).all(), "found NaN/Inf in MUSA output")
        self.assertTrue(np.isfinite(grad_musa_np).all(), "found NaN/Inf in MUSA grad")
        assert_allclose(
            y_musa_np, y_cpu_np,
            rtol=1e-6, atol=1e-6,
            msg="dim=6629 forward mismatch between MUSA and CPU"
        )
        assert_allclose(
            grad_musa_np, grad_cpu_np,
            rtol=5e-6, atol=1e-6,
            msg="dim=6629 backward mismatch between MUSA and CPU"
        )

        sum_axis = np.sum(grad_musa_np.astype(np.float64), axis=axis)
        np.testing.assert_allclose(
            sum_axis, np.zeros_like(sum_axis),
            rtol=0.0, atol=2e-5,
            err_msg="analytic property failed: sum(grad, axis) != 0"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
