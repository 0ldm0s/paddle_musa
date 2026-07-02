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


class TestCheckNumericsOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        self.assertIn("check_numerics", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["check_numerics"]))

    def test_check_all_finite_tensor(self):
        x = paddle.to_tensor(np.array([1.0, 2.0, 3.0], dtype="float32"))
        stats, values = paddle.amp.debugging.check_numerics(
            tensor=x,
            op_type="test_check_numerics",
            var_name="x",
            debug_mode=paddle.amp.debugging.DebugMode.CHECK_ALL,
        )
        np.testing.assert_array_equal(stats.numpy(), np.array([0, 0, 0], dtype="int64"))
        np.testing.assert_allclose(
            values.numpy(), np.array([3.0, 1.0, 2.0], dtype="float32"), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
