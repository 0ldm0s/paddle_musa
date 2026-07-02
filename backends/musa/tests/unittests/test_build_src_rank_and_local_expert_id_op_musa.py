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
from paddle.incubate.nn.functional import build_src_rank_and_local_expert_id


paddle.set_device("musa")


def build_expected(expert_num_global, num_local_experts):
    src_rank = []
    local_expert_id = []
    for expert_id, num_tokens in enumerate(expert_num_global):
        src_rank.extend([expert_id // num_local_experts] * num_tokens)
        local_expert_id.extend([expert_id % num_local_experts] * num_tokens)
    return np.array(src_rank, dtype="int32"), np.array(local_expert_id, dtype="int32")


class TestBuildSrcRankAndLocalExpertIdOpMusa(unittest.TestCase):
    def _run_case(self, expert_num_global, num_local_experts):
        expert_num_tensor = paddle.to_tensor(expert_num_global, dtype="int64")
        src_rank, local_expert_id = build_src_rank_and_local_expert_id(
            expert_num_tensor, expert_num_global, num_local_experts
        )

        expected_src_rank, expected_local_expert_id = build_expected(
            expert_num_global, num_local_experts
        )
        np.testing.assert_array_equal(src_rank.numpy(), expected_src_rank)
        np.testing.assert_array_equal(local_expert_id.numpy(), expected_local_expert_id)
        self.assertIn("musa", str(src_rank.place))
        self.assertIn("musa", str(local_expert_id.place))

    def test_even_experts(self):
        self._run_case([2, 0, 3, 1], 2)

    def test_multiple_ranks(self):
        self._run_case([1, 2, 0, 4, 3, 1], 3)


if __name__ == "__main__":
    unittest.main()
