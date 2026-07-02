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


class TestGridSampleGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("grid_sample_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["grid_sample_grad"]))

    def test_grid_sample_backward(self):
        paddle.set_device("musa")
        x_np = np.arange(9, dtype="float32").reshape([1, 1, 3, 3])
        grid_np = np.array([[[[-0.5, -0.5], [0.5, 0.5]]]], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        grid = paddle.to_tensor(grid_np, stop_gradient=False)
        out = paddle.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        out.sum().backward()
        self.assertEqual(list(x.grad.shape), [1, 1, 3, 3])
        self.assertEqual(list(grid.grad.shape), [1, 1, 2, 2])
        self.assertTrue(np.all(np.isfinite(x.grad.numpy())))
        self.assertTrue(np.all(np.isfinite(grid.grad.numpy())))


if __name__ == "__main__":
    unittest.main()
