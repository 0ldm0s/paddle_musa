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
from paddle.base import core


class TestMultiDotGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("multi_dot_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["multi_dot_grad"]))

    def test_multi_dot_backward(self):
        paddle.set_device("musa")
        a_np = np.arange(6, dtype="float32").reshape([2, 3]) / 10.0
        b_np = np.arange(12, dtype="float32").reshape([3, 4]) / 10.0
        c_np = np.arange(8, dtype="float32").reshape([4, 2]) / 10.0
        a = paddle.to_tensor(a_np, stop_gradient=False)
        b = paddle.to_tensor(b_np, stop_gradient=False)
        c = paddle.to_tensor(c_np, stop_gradient=False)
        y = paddle.linalg.multi_dot([a, b, c])
        y.sum().backward()
        ones = np.ones([2, 2], dtype="float32")
        np.testing.assert_allclose(a.grad.numpy(), ones @ (b_np @ c_np).T, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), a_np.T @ ones @ c_np.T, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(c.grad.numpy(), (a_np @ b_np).T @ ones, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
