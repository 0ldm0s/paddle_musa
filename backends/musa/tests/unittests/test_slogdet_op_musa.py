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


class TestSlogDetOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["slogdet", "slogdet_v2"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))

    def test_functional_slogdet_v2(self):
        paddle.set_device("musa")
        x_np = np.array([[1.0, 2.0], [3.0, 5.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        sign, logabsdet = paddle.linalg.slogdet(x)
        sign_np, logabsdet_np = np.linalg.slogdet(x_np)
        np.testing.assert_allclose(sign.numpy(), sign_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            logabsdet.numpy(), logabsdet_np, rtol=1e-5, atol=1e-5
        )
    def test_official_batched_slogdet_case(self):
        # Adapted from Paddle's determinant/slogdet official UT with a smaller
        # batched matrix shape for focused MUSA coverage.
        paddle.set_device("musa")
        x_np = np.array(
            [
                [[2.0, 1.0], [1.0, 3.0]],
                [[1.0, 2.0], [3.0, 5.0]],
                [[4.0, 1.0], [2.0, 2.0]],
            ],
            dtype="float32",
        )
        sign, logabsdet = paddle.linalg.slogdet(paddle.to_tensor(x_np))
        sign_np, logabsdet_np = np.linalg.slogdet(x_np)
        np.testing.assert_allclose(sign.numpy(), sign_np, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            logabsdet.numpy(), logabsdet_np, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
