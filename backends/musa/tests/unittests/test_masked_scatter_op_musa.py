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


class TestMaskedScatterOpMusa(unittest.TestCase):
    def test_masked_scatter(self):
        paddle.set_device("musa")
        x_np = np.arange(6, dtype="float32").reshape(2, 3)
        mask_np = np.array([[True, False, True], [False, True, False]])
        value_np = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")

        x = paddle.to_tensor(x_np)
        mask = paddle.to_tensor(mask_np)
        value = paddle.to_tensor(value_np)
        out = paddle.masked_scatter(x, mask, value)

        expected = x_np.copy()
        expected[mask_np] = value_np[: np.count_nonzero(mask_np)]
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_masked_scatter_broadcast_mask(self):
        paddle.set_device("musa")
        x_np = np.arange(6, dtype="float32").reshape(2, 3)
        mask_np = np.array([[True, False, True]])
        value_np = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")

        x = paddle.to_tensor(x_np)
        mask = paddle.to_tensor(mask_np)
        value = paddle.to_tensor(value_np)
        out = paddle.masked_scatter(x, mask, value)

        expanded_mask = np.broadcast_to(mask_np, x_np.shape)
        expected = x_np.copy()
        expected[expanded_mask] = value_np[: np.count_nonzero(expanded_mask)]
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_masked_scatter_grad(self):
        paddle.set_device("musa")
        x_np = np.arange(6, dtype="float32").reshape(2, 3)
        mask_np = np.array([[True, False, True], [False, True, False]])
        value_np = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        mask = paddle.to_tensor(mask_np)
        value = paddle.to_tensor(value_np, stop_gradient=False)
        out = paddle.masked_scatter(x, mask, value)
        out.sum().backward()

        expected_x_grad = np.where(mask_np, 0.0, 1.0).astype("float32")
        expected_value_grad = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")
        np.testing.assert_allclose(
            x.grad.numpy(), expected_x_grad, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            value.grad.numpy(), expected_value_grad, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
