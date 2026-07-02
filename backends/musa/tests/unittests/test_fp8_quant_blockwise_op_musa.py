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

import paddle
from paddle.base import core
from paddle.incubate.nn.functional import fp8


class TestFp8QuantBlockwiseOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("fp8_quant_blockwise", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["fp8_quant_blockwise"]))

    def test_official_unaligned_fp16_case(self):
        paddle.set_device("musa")
        paddle.seed(42)
        x = paddle.randn((129, 508), dtype=paddle.float16)
        x_q, scale = fp8.fp8_quant_blockwise(
            x,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
            using_pow2_scale=False,
            return_transpose_only=False,
            using_ue8m0_scale=False,
        )
        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(x_q.dtype, paddle.float8_e4m3fn)
        self.assertEqual(scale.dtype, paddle.float32)

        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=1)
        expanded_scale = expanded_scale[: x_q.shape[0], : x_q.shape[1]]
        x_qdq = x_q.astype("float32") * expanded_scale
        diff_squared = (x_qdq - x.astype("float32")) ** 2
        rmse = paddle.sqrt(paddle.sum(diff_squared) / x.numel())
        self.assertLessEqual(float(rmse.numpy()), 3e-2)


if __name__ == "__main__":
    unittest.main()
