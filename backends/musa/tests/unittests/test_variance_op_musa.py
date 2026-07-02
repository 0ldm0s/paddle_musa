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


class TestVarianceOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("variance", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["variance"]))

    def test_functional(self):
        paddle.set_device("musa")
        x_np = np.arange(12, dtype="float32").reshape([3, 4])
        x = paddle.to_tensor(x_np)
        out = paddle._C_ops.variance(x, [1], False)
        np.testing.assert_allclose(out.numpy(), np.var(x_np, axis=1), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
