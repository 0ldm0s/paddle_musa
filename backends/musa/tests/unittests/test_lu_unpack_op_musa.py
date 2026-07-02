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


class TestLUUnpackOpMusa(unittest.TestCase):
    def test_lu_unpack_and_grad(self):
        paddle.set_device("musa")
        x_np = np.array([[2.0, 3.0], [0.5, 4.0]], dtype="float32")
        pivots_np = np.array([1, 2], dtype="int32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        pivots = paddle.to_tensor(pivots_np)
        _, l, u = paddle.linalg.lu_unpack(x, pivots, True, False)

        expected_l = np.array([[1.0, 0.0], [0.5, 1.0]], dtype="float32")
        expected_u = np.array([[2.0, 3.0], [0.0, 4.0]], dtype="float32")
        np.testing.assert_allclose(l.numpy(), expected_l, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(u.numpy(), expected_u, rtol=1e-6, atol=1e-6)

        paddle.sum(l + u).backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones_like(x_np), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
