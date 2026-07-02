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


class TestIndexFillOpMusa(unittest.TestCase):
    def test_index_fill_and_grad(self):
        paddle.set_device("musa")
        x_np = np.arange(12, dtype="float32").reshape([3, 4])
        index_np = np.array([0, 2], dtype="int64")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        index = paddle.to_tensor(index_np)
        out = paddle.index_fill(x, index, axis=0, value=-3.0)

        expected = x_np.copy()
        expected[index_np, :] = -3.0
        np.testing.assert_allclose(out.numpy(), expected)

        paddle.sum(out).backward()
        expected_grad = np.ones_like(x_np)
        expected_grad[index_np, :] = 0.0
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_index_fill_with_int32_index_on_negative_axis(self):
        paddle.set_device("musa")
        x_np = np.arange(12, dtype="float32").reshape([3, 4])
        index_np = np.array([1, 3], dtype="int32")

        x = paddle.to_tensor(x_np)
        index = paddle.to_tensor(index_np)
        out = paddle.index_fill(x, index, axis=-1, value=9.0)

        expected = x_np.copy()
        expected[:, index_np] = 9.0
        np.testing.assert_allclose(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
