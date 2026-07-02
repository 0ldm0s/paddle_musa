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


class TestMulticlassNms3OpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        name = "multiclass_nms3"
        self.assertIn(name, kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels[name]))

    def test_official_small_nms_case(self):
        # Adapted from Paddle's multiclass_nms3 OpTest with a small deterministic
        # input: class 1 keeps only the highest-scoring overlapping box.
        paddle.set_device("musa")
        bboxes = paddle.to_tensor(
            [[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1], [2.0, 2.0, 3.0, 3.0]]],
            dtype="float32",
        )
        scores = paddle.to_tensor(
            [[[0.1, 0.1, 0.1], [0.9, 0.8, 0.2]]], dtype="float32"
        )
        out, index, nms_rois_num = paddle._C_ops.multiclass_nms3(
            bboxes,
            scores,
            None,
            0.3,
            3,
            3,
            0.3,
            True,
            1.0,
            0,
        )

        expected = np.array([[1.0, 0.9, 0.0, 0.0, 1.0, 1.0]], dtype="float32")
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(index.numpy(), np.array([[0]], dtype="int32"))
        np.testing.assert_array_equal(nms_rois_num.numpy(), np.array([1], dtype="int32"))


if __name__ == "__main__":
    unittest.main()
