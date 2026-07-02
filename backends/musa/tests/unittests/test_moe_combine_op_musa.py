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
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.nn.functional import moe_combine


class TestMoeCombineOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("moe_combine", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["moe_combine"]))

    def test_official_small_forward_and_backward_case(self):
        # Adapted from Paddle's incubate moe_combine UT: compare the fused op
        # with the embedding-based reference combine path, including gradients.
        paddle.set_device("musa")
        x = paddle.arange(1, 19, dtype="float32").reshape([6, 3])
        x.stop_gradient = False
        combine_weights = paddle.to_tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.5, 0.5]], dtype="float32"
        )
        combine_weights.stop_gradient = False
        scatter_index = paddle.to_tensor([[0, 1], [2, 3], [4, 5]], dtype="int32")
        grad = paddle.ones([3, 3], dtype="float32")

        out = moe_combine(x, combine_weights, scatter_index)
        paddle.autograd.backward([out], [grad], True)

        x_ref = paddle.arange(1, 19, dtype="float32").reshape([6, 3])
        x_ref.stop_gradient = False
        weights_ref = combine_weights.detach()
        weights_ref.stop_gradient = False
        gathered = F.embedding(scatter_index.astype("int64"), x_ref)
        out_ref = (weights_ref.unsqueeze(-1) * gathered).sum(1)
        paddle.autograd.backward([out_ref], [grad], True)

        np.testing.assert_allclose(out.numpy(), out_ref.numpy(), rtol=1e-6)
        np.testing.assert_allclose(x.grad.numpy(), x_ref.grad.numpy(), rtol=1e-6)
        self.assertEqual(combine_weights.grad.shape, [3, 2, 3])


if __name__ == "__main__":
    unittest.main()
