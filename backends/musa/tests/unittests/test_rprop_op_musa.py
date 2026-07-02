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


class TestRpropOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("rprop", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["rprop"]))

    def test_official_small_update_case(self):
        # Adapted from Paddle's rprop OpTest update formula with a focused
        # deterministic tensor that covers positive and negative grad*prev signs.
        paddle.set_device("musa")
        param = paddle.to_tensor([[0.1, -0.2], [0.3, -0.4]], dtype="float32")
        grad = paddle.to_tensor([[0.01, -0.02], [-0.03, 0.04]], dtype="float32")
        prev = paddle.to_tensor([[0.02, -0.01], [0.03, -0.04]], dtype="float32")
        learning_rate = paddle.to_tensor(
            [[0.002, 0.003], [0.004, 0.005]], dtype="float32"
        )
        learning_rate_range = paddle.to_tensor([0.001, 0.009], dtype="float32")
        etas = paddle.to_tensor([0.5, 1.2], dtype="float32")

        paddle._C_ops.rprop_(
            param,
            grad,
            prev,
            learning_rate,
            None,
            learning_rate_range,
            etas,
            False,
        )

        np.testing.assert_allclose(
            param.numpy(),
            np.array([[0.0976, -0.1964], [0.3, -0.4]], dtype="float32"),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            prev.numpy(),
            np.array([[0.01, -0.02], [0.0, 0.0]], dtype="float32"),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            learning_rate.numpy(),
            np.array([[0.0024, 0.0036], [0.002, 0.0025]], dtype="float32"),
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
