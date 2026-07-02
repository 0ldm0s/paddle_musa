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


class TestEmbeddingGradAddToOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("embedding_grad_add_to", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["embedding_grad_add_to"]))

    def test_official_small_equivalence_case(self):
        # Adapted from Paddle's incubate embedding_grad_add_to_ UT with reduced
        # dimensions and unique ids to keep the reference accumulation stable.
        paddle.set_device("musa")
        main_grad = paddle.zeros([8, 4], dtype="float32")
        dw_np = np.arange(12, dtype="float32").reshape([3, 4]) / 10.0
        dw = paddle.to_tensor(dw_np.astype("float32")).astype("bfloat16")
        ids = paddle.to_tensor([1, 3, 6], dtype="int32")

        fused_out = main_grad.detach().clone()
        paddle.incubate.nn.functional.embedding_grad_add_to_(ids, fused_out, dw)

        expected = np.zeros([8, 4], dtype="float32")
        expected[[1, 3, 6]] += dw.astype("float32").numpy()
        np.testing.assert_allclose(fused_out.numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
