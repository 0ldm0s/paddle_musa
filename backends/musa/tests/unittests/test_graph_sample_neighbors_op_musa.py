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


class TestGraphSampleNeighborsOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("graph_sample_neighbors", kernels)
        self.assertTrue(
            any("musa" in kernel for kernel in kernels["graph_sample_neighbors"])
        )

    def test_official_small_sample_result_case(self):
        # Adapted from Paddle's graph_sample_neighbors UT with a deterministic
        # CSR graph: sampled neighbors must come from the requested nodes.
        paddle.set_device("musa")
        row_np = np.array([1, 2, 0, 0, 1, 2], dtype="int64")
        colptr_np = np.array([0, 2, 3, 3, 6], dtype="int64")
        nodes_np = np.array([0, 1, 3], dtype="int64")
        dst_src_dict = {0: np.array([1, 2]), 1: np.array([0]), 3: np.array([0, 1, 2])}

        out_neighbors, out_count = paddle.incubate.graph_sample_neighbors(
            paddle.to_tensor(row_np),
            paddle.to_tensor(colptr_np),
            paddle.to_tensor(nodes_np),
            sample_size=2,
        )
        out_count_np = out_count.numpy()
        self.assertEqual(out_count_np.tolist(), [2, 1, 2])
        offset = 0
        for node, count in zip(nodes_np, out_count_np):
            neighbors = out_neighbors.numpy()[offset : offset + count]
            offset += count
            self.assertEqual(neighbors.shape[0], np.unique(neighbors).shape[0])
            self.assertTrue(np.isin(neighbors, dst_src_dict[int(node)]).all())


if __name__ == "__main__":
    unittest.main()
