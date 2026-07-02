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


class TestStridedSliceOpMusa(unittest.TestCase):
    def test_strided_slice_and_grad(self):
        paddle.set_device("musa")
        x_np = np.arange(12, dtype="float32").reshape([3, 4])
        x = paddle.to_tensor(x_np, stop_gradient=False)

        out = paddle.strided_slice(
            x, axes=[0, 1], starts=[0, 1], ends=[2, 4], strides=[1, 2]
        )
        expected = x_np[0:2:1, 1:4:2]
        np.testing.assert_allclose(out.numpy(), expected)

        paddle.sum(out).backward()
        expected_grad = np.zeros_like(x_np)
        expected_grad[0:2:1, 1:4:2] = 1.0
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)


if __name__ == "__main__":
    unittest.main()
