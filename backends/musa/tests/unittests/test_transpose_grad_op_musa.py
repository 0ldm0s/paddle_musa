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


class TestTransposeGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["transpose_grad", "trans_layout_grad"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))

    def test_functional_transpose_grad(self):
        paddle.set_device("musa")
        x_np = np.arange(24, dtype="float32").reshape([2, 3, 4])
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.transpose(x, perm=[1, 2, 0])
        y.sum().backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np.ones_like(x_np), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
