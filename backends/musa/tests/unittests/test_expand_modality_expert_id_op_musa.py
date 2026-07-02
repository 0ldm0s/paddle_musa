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
from paddle.incubate.nn.functional import expand_modality_expert_id


def expand_modality_expert_id_numpy(
    expert_id, num_expert_per_modality, group_size, modality_offset, is_group_expert
):
    out = expert_id.copy()
    seqlen, k = out.shape
    for i in range(seqlen):
        for j in range(k):
            e = out[i, j]
            if is_group_expert:
                e += j * group_size
            if num_expert_per_modality > 0:
                rank = e // num_expert_per_modality
                expert_id_in_rank = e % num_expert_per_modality
                e = (
                    rank * (num_expert_per_modality * 2)
                    + expert_id_in_rank
                    + modality_offset * num_expert_per_modality
                )
            out[i, j] = e
    return out


class TestExpandModalityExpertIdOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        self.assertIn("expand_modality_expert_id", kernels)
        self.assertTrue(
            any("musa" in kernel for kernel in kernels["expand_modality_expert_id"])
        )

    def test_dynamic_int64(self):
        expert_id_np = np.array([[0, 2, 4], [1, 3, 5]], dtype="int64")
        expert_id = paddle.to_tensor(expert_id_np)
        out = expand_modality_expert_id(
            expert_id,
            num_expert_per_modality=3,
            group_size=4,
            modality_offset=1,
            is_group_expert=True,
        )
        expected = expand_modality_expert_id_numpy(expert_id_np, 3, 4, 1, True)
        np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
