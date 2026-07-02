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


paddle.set_device("musa")


def get_c_embedding(start, end, table, ids):
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 0
    output = table[masked_input]
    output[input_mask] = 0.0
    return output.reshape((*ids.shape, table.shape[1]))


class TestCEmbeddingOpMusa(unittest.TestCase):
    def _run_forward_case(self, dtype, ids_dtype):
        np.random.seed(2026)
        table_np = np.random.random((17, 8)).astype("float32")
        ids_np = np.array([[1, 10, 12, 26], [9, 12, 20, 33]], dtype=ids_dtype)
        start_index = 10
        vocab_size = 34

        table = paddle.to_tensor(table_np, dtype=dtype)
        ids = paddle.to_tensor(ids_np)
        out = paddle._C_ops.c_embedding(table, ids, start_index, vocab_size)
        expected = get_c_embedding(
            start_index, start_index + table_np.shape[0], table_np, ids_np
        )

        atol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-6
        rtol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-6
        np.testing.assert_allclose(
            out.astype("float32").numpy(), expected, rtol=rtol, atol=atol
        )
        self.assertIn("musa", str(out.place))

    def test_forward_float32_int64(self):
        self._run_forward_case("float32", "int64")

    def test_forward_float32_int32(self):
        self._run_forward_case("float32", "int32")

    def test_forward_float16(self):
        self._run_forward_case("float16", "int32")

    def test_forward_bfloat16(self):
        self._run_forward_case("bfloat16", "int32")

    def test_grad_float32(self):
        table_np = np.random.random((17, 8)).astype("float32")
        ids_np = np.array([[10, 12, 12], [11, 20, 9]], dtype="int64")
        table = paddle.to_tensor(table_np, stop_gradient=False)
        ids = paddle.to_tensor(ids_np)
        out = paddle._C_ops.c_embedding(table, ids, 10, 34)
        loss = paddle.sum(out)
        loss.backward()

        expected_grad = np.zeros_like(table_np)
        for idx in ids_np.flatten():
            if 10 <= idx < 27:
                expected_grad[idx - 10] += 1.0
        np.testing.assert_allclose(table.grad.numpy(), expected_grad)


if __name__ == "__main__":
    unittest.main()
