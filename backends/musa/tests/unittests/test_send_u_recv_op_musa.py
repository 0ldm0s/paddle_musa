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


class TestSendURecvOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("send_u_recv", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["send_u_recv"]))

    def test_functional_sum(self):
        paddle.set_device("musa")
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"
        )
        src_index = paddle.to_tensor([0, 1, 2], dtype="int64")
        dst_index = paddle.to_tensor([1, 0, 1], dtype="int64")
        out = paddle.geometric.send_u_recv(
            x, src_index, dst_index, reduce_op="sum", out_size=2
        )
        expected = np.array([[3.0, 4.0], [6.0, 8.0]], dtype="float32")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
