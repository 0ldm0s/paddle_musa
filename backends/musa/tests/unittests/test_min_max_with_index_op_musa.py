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


def _max_with_index(x, dim, keepdim=False):
    values, indices = paddle._C_ops.max_with_index(x, dim, keepdim, False)
    indices.stop_gradient = True
    return values, indices


def _min_with_index(x, dim, keepdim=False):
    values, indices = paddle._C_ops.min_with_index(x, dim, keepdim, False)
    indices.stop_gradient = True
    return values, indices


class TestMinMaxWithIndexOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_max_with_index_and_grad(self):
        x_np = np.array(
            [[[1.0, 3.0, -2.0], [4.0, 0.0, 5.0]], [[2.0, -1.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype="float32",
        )
        x = paddle.to_tensor(x_np, stop_gradient=False)
        values, indices = _max_with_index(x, dim=1, keepdim=False)

        np.testing.assert_allclose(values.numpy(), np.max(x_np, axis=1))
        np.testing.assert_array_equal(indices.numpy(), np.argmax(x_np, axis=1).astype("int64"))

        paddle.sum(values).backward()
        expected_grad = np.zeros_like(x_np)
        np.put_along_axis(
            expected_grad,
            np.expand_dims(np.argmax(x_np, axis=1), axis=1),
            1.0,
            axis=1,
        )
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_min_with_index_keepdim_and_grad(self):
        x_np = np.array(
            [[[1.0, 3.0, -2.0], [4.0, 0.0, 5.0]], [[2.0, -1.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype="float32",
        )
        x = paddle.to_tensor(x_np, stop_gradient=False)
        values, indices = _min_with_index(x, dim=-1, keepdim=True)

        np.testing.assert_allclose(values.numpy(), np.min(x_np, axis=-1, keepdims=True))
        np.testing.assert_array_equal(
            indices.numpy(), np.argmin(x_np, axis=-1).astype("int64")[..., None]
        )

        paddle.sum(values).backward()
        expected_grad = np.zeros_like(x_np)
        np.put_along_axis(
            expected_grad,
            np.argmin(x_np, axis=-1)[..., None],
            1.0,
            axis=-1,
        )
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_max_with_index_uint8(self):
        x_np = np.array([[1, 9, 3], [7, 2, 5]], dtype="uint8")
        x = paddle.to_tensor(x_np)
        values, indices = _max_with_index(x, dim=1, keepdim=True)

        np.testing.assert_array_equal(values.numpy(), np.max(x_np, axis=1, keepdims=True))
        np.testing.assert_array_equal(
            indices.numpy(), np.argmax(x_np, axis=1).astype("int64")[:, None]
        )


if __name__ == "__main__":
    unittest.main()
