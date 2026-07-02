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


class TestCholeskyGradOpMusa(unittest.TestCase):
    def test_cholesky_grad_matches_cpu(self):
        out_np = np.array([[2.0, 0.0], [0.5, 1.5]], dtype="float32")
        out_grad_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")

        paddle.set_device("cpu")
        expected = paddle._C_ops.cholesky_grad(
            paddle.to_tensor(out_np), paddle.to_tensor(out_grad_np), False
        ).numpy()

        paddle.set_device("musa")
        actual = paddle._C_ops.cholesky_grad(
            paddle.to_tensor(out_np), paddle.to_tensor(out_grad_np), False
        ).numpy()

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
