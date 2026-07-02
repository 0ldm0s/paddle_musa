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
from paddle.incubate.nn.functional import moe_gate_dispatch_partial_nosoftmaxtopk


class TestMoeGateDispatchPartialNoSoftmaxTopkOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        name = "moe_gate_dispatch_partial_nosoftmaxtopk"
        self.assertIn(name, kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels[name]))

    def test_official_reverse_token_drop_case(self):
        # Adapted from Paddle's incubate partial nosoftmax topk UT.
        paddle.set_device("musa")
        s, e, d = 3, 4, 3
        k = 2
        capacity = 2
        x = (paddle.arange(s) + 1).unsqueeze(-1).expand([s, d]).astype("bfloat16")
        combine_weights = paddle.randn([s, k])
        expert_id = paddle.to_tensor([[0, 1], [0, 1], [0, 2]], dtype="int32")

        out, _, _, _, _, num_expert_local = moe_gate_dispatch_partial_nosoftmaxtopk(
            x,
            combine_weights,
            expert_id,
            k,
            capacity,
            e,
            False,
            0,
            2,
            reverse_token_drop=True,
        )

        y0, y1 = out.split([i for i in num_expert_local.tolist() if i > 0])
        self.assertEqual(y0[:, 0].astype("int32").tolist(), [2, 3])
        self.assertEqual(y1[:, 0].astype("int32").tolist(), [1, 2])

    def test_official_empty_output_case(self):
        paddle.set_device("musa")
        s, e, d = 3, 4, 3
        k = 2
        capacity = 2
        x = (paddle.arange(s) + 1).unsqueeze(-1).expand([s, d]).astype("bfloat16")
        combine_weights = paddle.randn([s, k])
        expert_id = paddle.to_tensor([[0, 1], [0, 1], [0, 2]], dtype="int32")

        _, _, _, _, _, num_expert_local = moe_gate_dispatch_partial_nosoftmaxtopk(
            x,
            combine_weights,
            expert_id,
            k,
            capacity,
            e,
            False,
            3,
            4,
            reverse_token_drop=True,
        )
        self.assertTrue(all(i == 0 for i in num_expert_local.tolist()))


if __name__ == "__main__":
    unittest.main()
