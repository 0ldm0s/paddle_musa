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
from paddle.base import core
from paddle.incubate.nn.functional.moe_gate_dispatch_and_quant import (
    math_moe_gate_dispatch_and_quant,
    moe_gate_dispatch_and_quant,
)


class TestMoeGateDispatchAndQuantOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("moe_gate_dispatch_and_quant", kernels)
        self.assertTrue(
            any("musa" in kernel for kernel in kernels["moe_gate_dispatch_and_quant"])
        )

    def test_official_small_math_reference_case(self):
        # Adapted from Paddle's incubate FP8 MoE dispatch+quant UT with a
        # reduced shape for MUSA CI: compare fused outputs with math reference.
        paddle.set_device("musa")
        paddle.seed(42)
        x = paddle.randn([8, 128], dtype="bfloat16")
        gate_logits = paddle.randn([8, 4], dtype="float32")

        out = moe_gate_dispatch_and_quant(
            x,
            gate_logits,
            corr_bias=None,
            k=2,
            capacity=4,
            use_pad=True,
            use_pow2_scale=False,
        )
        ref = math_moe_gate_dispatch_and_quant(
            x,
            gate_logits,
            corr_bias=None,
            k=2,
            capacity=4,
            use_pad=True,
            use_pow2_scale=False,
        )

        for actual, expected in zip(out[2:], ref[2:]):
            np.testing.assert_equal(actual._md5sum(), expected._md5sum())
        np.testing.assert_equal(out[0].shape, ref[0].shape)
        np.testing.assert_equal(out[1].shape, ref[1].shape)
        np.testing.assert_equal(out[1]._md5sum(), ref[1]._md5sum())
        np.testing.assert_equal(
            out[0].astype("float32")._md5sum(), ref[0].astype("float32")._md5sum()
        )


if __name__ == "__main__":
    unittest.main()
