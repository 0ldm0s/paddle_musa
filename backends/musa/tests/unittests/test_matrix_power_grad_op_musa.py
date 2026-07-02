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


class TestMatrixPowerGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("matrix_power_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["matrix_power_grad"]))

    def test_matrix_power_backward(self):
        paddle.set_device("musa")
        x_np = np.array([[1.0, 2.0], [3.0, 5.0]], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.linalg.matrix_power(x, 2)
        y.sum().backward()
        expected = np.array(
            [
                [x_np[:, 0].sum() + x_np[0, :].sum(), x_np[:, 0].sum() + x_np[1, :].sum()],
                [x_np[:, 1].sum() + x_np[0, :].sum(), x_np[:, 1].sum() + x_np[1, :].sum()],
            ],
            dtype="float32",
        )
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
