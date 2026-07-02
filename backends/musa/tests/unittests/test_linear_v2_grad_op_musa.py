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


class TestLinearV2GradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("linear_v2_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["linear_v2_grad"]))

    def test_linear_backward(self):
        paddle.set_device("musa")
        x_np = np.arange(6, dtype="float32").reshape([2, 3]) / 10.0
        w_np = np.arange(12, dtype="float32").reshape([3, 4]) / 10.0
        b_np = np.arange(4, dtype="float32") / 10.0
        x = paddle.to_tensor(x_np, stop_gradient=False)
        w = paddle.to_tensor(w_np, stop_gradient=False)
        b = paddle.to_tensor(b_np, stop_gradient=False)
        y = paddle.nn.functional.linear(x, w, b)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones([2, 4], dtype="float32") @ w_np.T, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(w.grad.numpy(), x_np.T @ np.ones([2, 4], dtype="float32"), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), np.full([4], 2.0, dtype="float32"), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
