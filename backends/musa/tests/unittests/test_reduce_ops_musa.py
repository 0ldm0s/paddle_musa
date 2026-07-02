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


class TestReduceOpsMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_max(self):
        x_np = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        out = paddle.max(x, axis=1, keepdim=True)
        np.testing.assert_allclose(out.numpy(), np.max(x_np, axis=1, keepdims=True))

    def test_prod(self):
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        out = paddle.prod(x, axis=0, keepdim=False)
        np.testing.assert_allclose(out.numpy(), np.prod(x_np, axis=0))

    def test_any(self):
        x_np = np.array([[False, False, True], [False, False, False]], dtype="bool")
        x = paddle.to_tensor(x_np)
        out = paddle.any(x, axis=1, keepdim=False)
        np.testing.assert_array_equal(out.numpy(), np.any(x_np, axis=1))

    def test_nansum(self):
        x_np = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype="float32")
        x = paddle.to_tensor(x_np)
        out = paddle.nansum(x, axis=1, keepdim=True)
        np.testing.assert_allclose(out.numpy(), np.nansum(x_np, axis=1, keepdims=True))


if __name__ == "__main__":
    unittest.main()
