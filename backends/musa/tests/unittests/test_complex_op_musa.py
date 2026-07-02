# Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle


paddle.set_device("musa")


class TestComplexOpMusa(unittest.TestCase):
    def test_complex_real_imag_conj_forward(self):
        real_np = np.array([[1.0, -2.0], [3.5, -4.5]], dtype="float32")
        imag_np = np.array([[0.5, -1.5], [2.0, -3.0]], dtype="float32")
        real = paddle.to_tensor(real_np)
        imag = paddle.to_tensor(imag_np)

        out = paddle.complex(real, imag)
        expected = real_np.astype("complex64") + 1j * imag_np.astype("complex64")
        np.testing.assert_allclose(out.numpy(), expected)
        self.assertIn("musa", str(out.place))

        np.testing.assert_allclose(paddle.real(out).numpy(), real_np)
        np.testing.assert_allclose(paddle.imag(out).numpy(), imag_np)
        np.testing.assert_allclose(paddle.conj(out).numpy(), np.conj(expected))

    def test_complex_grad(self):
        real_np = np.array([[1.0, -2.0], [3.5, -4.5]], dtype="float32")
        imag_np = np.array([[0.5, -1.5], [2.0, -3.0]], dtype="float32")
        real = paddle.to_tensor(real_np, stop_gradient=False)
        imag = paddle.to_tensor(imag_np, stop_gradient=False)

        out = paddle.complex(real, imag)
        loss = paddle.sum(paddle.real(out) + paddle.imag(out))
        loss.backward()

        np.testing.assert_allclose(real.grad.numpy(), np.ones_like(real_np))
        np.testing.assert_allclose(imag.grad.numpy(), np.ones_like(imag_np))

    def test_real_imag_grad(self):
        x_np = np.array([[1.0 + 2.0j, -3.0 + 4.0j]], dtype="complex64")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        paddle.sum(paddle.real(x)).backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones_like(x_np).real + 0j)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        paddle.sum(paddle.imag(x)).backward()
        np.testing.assert_allclose(x.grad.numpy(), 1j * np.ones_like(x_np))

    def test_conj_float_and_complex(self):
        float_np = np.array([1.0, -2.0, 3.0], dtype="float32")
        float_out = paddle.conj(paddle.to_tensor(float_np))
        np.testing.assert_allclose(float_out.numpy(), float_np)

        complex_np = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype="complex64")
        complex_out = paddle.conj(paddle.to_tensor(complex_np))
        np.testing.assert_allclose(complex_out.numpy(), np.conj(complex_np))


if __name__ == "__main__":
    unittest.main()
