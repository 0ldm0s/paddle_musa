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


class TestTriangularSolveOpMusa(unittest.TestCase):
    def test_triangular_solve_and_grad(self):
        paddle.set_device("musa")
        x_np = np.array([[2.0, 0.0], [0.5, 1.5]], dtype="float32")
        y_np = np.array([[1.0], [3.5]], dtype="float32")

        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)
        out = paddle.linalg.triangular_solve(x, y, upper=False)

        expected = np.linalg.solve(np.tril(x_np), y_np)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

        paddle.sum(out).backward()
        grad_out = np.ones_like(expected)
        expected_y_grad = np.linalg.solve(np.tril(x_np).T, grad_out)
        expected_x_grad = -expected_y_grad @ expected.T
        expected_x_grad = np.tril(expected_x_grad)
        np.testing.assert_allclose(y.grad.numpy(), expected_y_grad, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x.grad.numpy(), expected_x_grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
