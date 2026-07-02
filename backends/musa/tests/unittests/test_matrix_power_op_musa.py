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


class TestMatrixPowerOpMusa(unittest.TestCase):
    def test_matrix_power(self):
        paddle.set_device("musa")
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        out = paddle.linalg.matrix_power(x, 3)
        np.testing.assert_allclose(out.numpy(), np.linalg.matrix_power(x_np, 3), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
