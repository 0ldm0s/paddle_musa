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


class TestRandomGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("random_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["random_grad"]))

    def test_official_small_zero_grad_case(self):
        # Adapted from Paddle's random_ grad UT with a reduced tensor shape:
        # in-place randomization should stop gradients for both random_ forms.
        paddle.set_device("musa")
        for random_args in [(), (0, 10)]:
            x = paddle.ones([8, 16], dtype="float32")
            x.stop_gradient = False
            y = x * 0.5
            y.retain_grads()
            y.random_(*random_args)
            y.sum().backward()
            np.testing.assert_array_equal(y.grad.numpy(), np.zeros([8, 16], dtype="float32"))


if __name__ == "__main__":
    unittest.main()
