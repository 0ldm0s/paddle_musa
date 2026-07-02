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


class TestMaskedSelectGradOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("masked_select_grad", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["masked_select_grad"]))

    def test_masked_select_backward(self):
        paddle.set_device("musa")
        x_np = np.arange(6, dtype="float32").reshape([2, 3])
        mask_np = np.array([[True, False, True], [False, True, False]])
        x = paddle.to_tensor(x_np, stop_gradient=False)
        mask = paddle.to_tensor(mask_np)
        y = paddle.masked_select(x, mask)
        y.sum().backward()
        np.testing.assert_allclose(
            x.grad.numpy(), mask_np.astype("float32"), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
