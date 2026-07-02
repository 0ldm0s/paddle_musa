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
from paddle.incubate.nn.functional import batched_gemm


class TestBatchedGemmOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_m_grouped_gemm(self):
        lhs_np = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        rhs_np = np.array(
            [[[1.0, 0.5], [0.0, 2.0]], [[2.0, 1.0], [1.5, 0.0]]],
            dtype="float32",
        )
        batch_sizes = [2, 1]

        lhs = paddle.to_tensor(lhs_np)
        rhs = paddle.to_tensor(rhs_np)
        out = batched_gemm(lhs, rhs, batch_sizes, False, False)

        expected = np.concatenate(
            [lhs_np[:2] @ rhs_np[0], lhs_np[2:] @ rhs_np[1]], axis=0
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_m_grouped_gemm_trans_rhs(self):
        lhs_np = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        rhs_base = np.array(
            [[[1.0, 0.0], [0.5, 2.0]], [[2.0, 1.5], [1.0, 0.0]]],
            dtype="float32",
        )
        rhs_np = np.swapaxes(rhs_base, -1, -2)
        batch_sizes = [2, 1]

        lhs = paddle.to_tensor(lhs_np)
        rhs = paddle.to_tensor(rhs_np)
        out = batched_gemm(lhs, rhs, batch_sizes, False, True)

        expected = np.concatenate(
            [lhs_np[:2] @ rhs_np[0].T, lhs_np[2:] @ rhs_np[1].T], axis=0
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_k_grouped_gemm(self):
        lhs_np = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        rhs_np = np.array(
            [[1.0, 0.5, 2.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0]],
            dtype="float32",
        )
        batch_sizes = [2, 1]

        lhs = paddle.to_tensor(lhs_np)
        rhs = paddle.to_tensor(rhs_np)
        out = batched_gemm(lhs, rhs, batch_sizes, True, False)

        expected = np.stack(
            [lhs_np[:2].T @ rhs_np[:2], lhs_np[2:].T @ rhs_np[2:]], axis=0
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
