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


def class_center_sample_numpy(label, num_classes, num_samples):
    unique_label = np.unique(label)
    remapped = {v: i for i, v in enumerate(unique_label)}
    remapped_label = np.array([remapped[v] for v in label], dtype=label.dtype)
    return remapped_label, unique_label.astype(label.dtype)


class TestClassCenterSampleOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        self.assertIn("class_center_sample", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["class_center_sample"]))

    def test_dynamic_int64(self):
        label_np = np.array([3, 1, 3, 5, 1, 7], dtype="int64")
        label = paddle.to_tensor(label_np)
        remapped_label, sampled_class_center = paddle.nn.functional.class_center_sample(
            label, num_classes=10, num_samples=4
        )
        expected_remapped, expected_center_prefix = class_center_sample_numpy(
            label_np, 10, 4
        )
        np.testing.assert_array_equal(remapped_label.numpy(), expected_remapped)
        np.testing.assert_array_equal(
            sampled_class_center.numpy()[: len(expected_center_prefix)],
            expected_center_prefix,
        )

    def test_dynamic_int32(self):
        label_np = np.array([2, 2, 0, 4, 6], dtype="int32")
        label = paddle.to_tensor(label_np)
        remapped_label, sampled_class_center = paddle.nn.functional.class_center_sample(
            label, num_classes=8, num_samples=3
        )
        expected_remapped, expected_center_prefix = class_center_sample_numpy(
            label_np, 8, 3
        )
        np.testing.assert_array_equal(remapped_label.numpy(), expected_remapped)
        np.testing.assert_array_equal(
            sampled_class_center.numpy()[: len(expected_center_prefix)],
            expected_center_prefix,
        )


if __name__ == "__main__":
    unittest.main()
