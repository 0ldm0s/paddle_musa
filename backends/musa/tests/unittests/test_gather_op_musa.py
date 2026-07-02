# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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


paddle.set_device("musa")


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    out = x_transpose[index, ...]
    return np.swapaxes(out, 0, axis)


def gather_grad_numpy(x_shape, index, axis, grad_out):
    grad = np.zeros(x_shape, dtype=grad_out.dtype)
    grad_t = np.swapaxes(grad, 0, axis)
    grad_out_t = np.swapaxes(grad_out, 0, axis)
    for out_i, x_i in enumerate(index):
        grad_t[x_i] += grad_out_t[out_i]
    return np.swapaxes(grad_t, 0, axis)


class TestGatherMusa(unittest.TestCase):
    def check_case(self, x_shape, index, axis, dtype="float32", index_dtype="int64"):
        np.random.seed(2025)
        x_np = np.random.random(x_shape).astype(dtype)
        index_np = np.array(index).astype(index_dtype)
        expected = gather_numpy(x_np, index_np, axis)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        index_tensor = paddle.to_tensor(index_np)
        out = paddle.gather(x, index_tensor, axis=axis)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-4, atol=1e-5)

        grad_out_np = np.random.random(expected.shape).astype(dtype)
        grad_out = paddle.to_tensor(grad_out_np)
        (x_grad,) = paddle.grad(out, x, grad_outputs=grad_out)
        expected_grad = gather_grad_numpy(x_np.shape, index_np, axis, grad_out_np)
        np.testing.assert_allclose(x_grad.numpy(), expected_grad, rtol=1e-4, atol=1e-5)

    def test_gather_axis0_int32(self):
        self.check_case((10, 20), [1, 3, 5], 0, "float32", "int32")

    def test_gather_axis0_int64(self):
        self.check_case((10, 20), [1, 3, 5], 0, "float32", "int64")

    def test_gather_axis1(self):
        self.check_case((3, 88, 3), [1, 3, 5], 1, "float32", "int64")

    def test_gather_axis2(self):
        self.check_case((10, 8, 10), [1, 3, 5], 2, "float32", "int64")

    def test_gather_duplicate_index_grad(self):
        self.check_case((10, 20), [1, 1, 3], 0, "float32", "int32")

    def test_gather_float16(self):
        self.check_case((10, 20), [1, 3, 5], 0, "float16", "int64")

    def test_gather_negative_axis(self):
        self.check_case((10, 8, 10), [1, 3, 5], -1, "float32", "int64")


class TestGatherAPIMusa(unittest.TestCase):
    def test_empty_index(self):
        x = paddle.to_tensor([[1, 2], [3, 4]])
        index = paddle.to_tensor(np.array([]).astype("int64"))
        for axis in range(len(x.shape)):
            out = paddle.gather(x, index, axis=axis)
            expected_shape = list(x.shape)
            expected_shape[axis] = 0
            self.assertEqual(list(out.shape), expected_shape)

    def test_out_argument_backward(self):
        x = paddle.arange(12, dtype=paddle.float32).reshape([3, 4])
        index = paddle.to_tensor([0, 1, 1], dtype=paddle.int64)
        x.stop_gradient = False
        res_out = paddle.to_tensor(0)
        res = paddle.gather(x, axis=1, index=index, out=res_out)
        expected = np.array(
            [[0.0, 1.0, 1.0], [4.0, 5.0, 5.0], [8.0, 9.0, 9.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(res.numpy(), expected)
        np.testing.assert_allclose(res_out.numpy(), expected)
        res.backward()
        expected_grad = np.array(
            [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)


if __name__ == "__main__":
    unittest.main()
