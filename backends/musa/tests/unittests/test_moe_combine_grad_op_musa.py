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


class TestMoeCombineGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["moe_combine_grad", "moe_combine_auto_grad"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))

    def test_official_small_grad_case(self):
        # Adapted from Paddle's MoE combine PyLayer backward: call
        # moe_combine_grad and reduce grad_combine_weight_helper over hidden dim.
        paddle.set_device("musa")
        x = paddle.arange(1, 19, dtype="float32").reshape([6, 3])
        combine_weights = paddle.to_tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.5, 0.5]], dtype="float32"
        )
        scatter_index = paddle.to_tensor([[0, 1], [2, 3], [4, 5]], dtype="int32")
        grad_y = paddle.ones([3, 3], dtype="float32")

        grad_x, grad_weight_helper = paddle._C_ops.moe_combine_grad(
            x, combine_weights, scatter_index, grad_y
        )
        grad_weight = grad_weight_helper.sum(-1).reshape(combine_weights.shape)

        expected_grad_x = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype="float32",
        )
        expected_grad_weight = np.array(
            [[6.0, 15.0], [24.0, 33.0], [42.0, 51.0]], dtype="float32"
        )
        np.testing.assert_allclose(grad_x.numpy(), expected_grad_x, rtol=1e-6)
        np.testing.assert_allclose(grad_weight.numpy(), expected_grad_weight, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
