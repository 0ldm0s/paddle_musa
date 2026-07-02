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


class TestRandomOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("random", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["random"]))

    def test_functional_int64(self):
        paddle.set_device("musa")
        x = paddle.empty([4, 5], dtype="int64")
        out = paddle.randint_like(x, low=2, high=7)
        self.assertEqual(list(out.shape), [4, 5])
        self.assertEqual(out.dtype, paddle.int64)
        out_np = out.numpy()
        self.assertTrue(np.all(out_np >= 2))
        self.assertTrue(np.all(out_np < 7))


if __name__ == "__main__":
    unittest.main()
