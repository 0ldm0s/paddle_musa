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

import math
import unittest

import numpy as np
import paddle
from paddle.base import core


class TestGammainccGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("gammaincc_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["gammaincc_grad"]))

    def test_official_small_y_grad_case(self):
        # Adapted from Paddle's gammaincc grad coverage: validate d/dy of
        # gammaincc(a, y), which is -exp(-y) * y ** (a - 1) / gamma(a).
        paddle.set_device("musa")
        x_np = np.array([1.5, 2.0, 3.0], dtype="float32")
        y_np = np.array([0.5, 1.0, 2.0], dtype="float32")
        x = paddle.to_tensor(x_np)
        x.stop_gradient = True
        y = paddle.to_tensor(y_np)
        y.stop_gradient = False

        out = paddle.gammaincc(x, y)
        out.sum().backward()

        expected = -np.exp(-y_np) * np.power(y_np, x_np - 1) / np.array(
            [math.gamma(float(v)) for v in x_np], dtype="float32"
        )
        np.testing.assert_allclose(y.grad.numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
