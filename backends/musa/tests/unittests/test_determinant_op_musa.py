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


class TestDeterminantOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        self.assertIn("determinant", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["determinant"]))

    def test_dynamic_float32(self):
        x_np = np.array([[2.0, 1.0], [3.0, 4.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        out = paddle.linalg.det(x)
        np.testing.assert_allclose(out.numpy(), np.linalg.det(x_np), rtol=1e-5, atol=1e-5)

    def test_dynamic_batched_float64(self):
        x_np = np.array(
            [
                [[1.0, 2.0], [3.0, 5.0]],
                [[2.0, 0.0], [1.0, 3.0]],
            ],
            dtype="float64",
        )
        x = paddle.to_tensor(x_np)
        out = paddle.linalg.det(x)
        np.testing.assert_allclose(out.numpy(), np.linalg.det(x_np), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
