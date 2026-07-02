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
import paddle.incubate.nn.functional as F
from paddle import _C_ops
from paddle.base import core


class TestCrossEntropyBwdWDowncastOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        self.assertIn("cross_entropy_with_softmax_bwd_w_downcast", kernels)
        self.assertTrue(
            any(
                "musa" in kernel
                for kernel in kernels["cross_entropy_with_softmax_bwd_w_downcast"]
            )
        )

    def test_dynamic_grad_matches_reference(self):
        preds_np = np.array(
            [[[0.2, 1.0, -0.5, 0.3], [1.2, -0.7, 0.4, 0.0]]], dtype="float32"
        )
        labels_np = np.array([[[1], [2]]], dtype="int64")
        preds = paddle.to_tensor(preds_np, stop_gradient=False)
        labels = paddle.to_tensor(labels_np)
        softmax, loss = _C_ops.cross_entropy_with_softmax(
            preds, labels, False, True, False, -100, -1
        )
        loss_grad = paddle.ones_like(loss)
        custom_grad = F.cross_entropy_with_softmax_bwd_w_downcast(
            labels, softmax, loss_grad
        )
        reference_grad = _C_ops.cross_entropy_with_softmax_grad(
            labels, softmax, loss_grad, False, True, False, -100, -1
        )
        self.assertEqual(custom_grad.dtype, paddle.bfloat16)
        np.testing.assert_allclose(
            custom_grad.astype("float32").numpy(),
            reference_grad.numpy(),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
