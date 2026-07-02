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


class TestSliceOpMusa(unittest.TestCase):
    def test_slice_and_grad(self):
        paddle.set_device("musa")
        x_np = np.arange(12, dtype="float32").reshape([3, 4])
        x = paddle.to_tensor(x_np, stop_gradient=False)

        out = paddle.slice(x, axes=[0, 1], starts=[1, 0], ends=[3, 3])
        expected = x_np[1:3, 0:3]
        np.testing.assert_allclose(out.numpy(), expected)

        paddle.sum(out).backward()
        expected_grad = np.zeros_like(x_np)
        expected_grad[1:3, 0:3] = 1.0
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_slice_array_grad_kernels_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["slice_array_grad", "slice_array_dense_grad"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]), kernels[op_name])


if __name__ == "__main__":
    unittest.main()
