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
from paddle.tensor import random


class TestUniformRandomBatchSizeLikeOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("uniform_random_batch_size_like", kernels)
        self.assertTrue(
            any("musa" in kernel for kernel in kernels["uniform_random_batch_size_like"])
        )

    def test_functional(self):
        paddle.set_device("musa")
        x = paddle.empty([2, 3], dtype="float32")
        out = random.uniform_random_batch_size_like(
            x, shape=[-1, 4], dtype="float32", min=0.0, max=1.0, seed=123
        )
        self.assertEqual(list(out.shape), [2, 4])
        self.assertEqual(out.dtype, paddle.float32)
        out_np = out.numpy()
        self.assertTrue(np.all(out_np >= 0.0))
        self.assertTrue(np.all(out_np <= 1.0))


if __name__ == "__main__":
    unittest.main()
