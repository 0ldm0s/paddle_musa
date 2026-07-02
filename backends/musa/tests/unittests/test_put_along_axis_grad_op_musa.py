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


class TestPutAlongAxisGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("put_along_axis_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["put_along_axis_grad"]))

    def test_official_mul_grad_zero_guard_cases(self):
        # Adapted from Paddle's put_along_axis mul backward zero-guard UT:
        # gradients should stay finite when x or value contains zero.
        paddle.set_device("musa")
        x = paddle.to_tensor(
            [[[1.0, 0.0, 3.0], [0.0, 5.0, 6.0]]], dtype="float32"
        )
        x.stop_gradient = False
        index = paddle.to_tensor([[[0, 1, 0], [1, 0, 1]]], dtype="int64")
        value = paddle.to_tensor(
            [[[2.0, 1.0, 4.0], [1.0, 3.0, 5.0]]], dtype="float32"
        )
        value.stop_gradient = True
        out = paddle.put_along_axis(
            x, index, value, axis=2, reduce="mul", include_self=True
        )
        out.sum().backward()
        self.assertTrue(np.isfinite(x.grad.numpy()).all())

        x = paddle.to_tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype="float32"
        )
        x.stop_gradient = True
        value = paddle.to_tensor(
            [[[2.0, 0.0, 4.0], [0.0, 3.0, 5.0]]], dtype="float32"
        )
        value.stop_gradient = False
        out = paddle.put_along_axis(
            x, index, value, axis=2, reduce="mul", include_self=True
        )
        out.sum().backward()
        self.assertTrue(np.isfinite(value.grad.numpy()).all())


if __name__ == "__main__":
    unittest.main()
