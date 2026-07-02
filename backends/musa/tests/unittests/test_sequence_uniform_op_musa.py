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

import sys
import unittest
from pathlib import Path

import numpy as np
import paddle

sys.path.append(str(Path(__file__).resolve().parents[4] / "Paddle" / "test" / "legacy_test"))
from op_test import OpTest
from paddle.base import core
from paddle.base.framework import convert_np_dtype_to_proto_type


def get_places(self):
    return [paddle.CustomPlace("musa", 0)]


def stable_softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


OpTest._get_places = get_places


class SequenceMaskOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "sequence_mask"
        self.x = np.array([[0, 3, 4], [5, 7, 9]])
        self.maxlen = 10
        self.mask_dtype = "int64"
        self.inputs = {"X": self.x}
        self.outputs = {"Y": self.calc_ground_truth_mask()}
        self.attrs = {
            "maxlen": self.maxlen,
            "out_dtype": convert_np_dtype_to_proto_type(self.mask_dtype),
        }

    def calc_ground_truth_mask(self):
        maxlen = np.max(self.x) if self.maxlen < 0 else self.maxlen
        shape = (*self.x.shape, maxlen)
        index_broadcast = np.broadcast_to(
            np.reshape(range(maxlen), newshape=[1] * self.x.ndim + [-1]), shape=shape
        )
        x_broadcast = np.broadcast_to(
            np.reshape(self.x, newshape=(*self.x.shape, -1)), shape=shape
        )
        return (index_broadcast < x_broadcast).astype(self.mask_dtype)

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class SequenceMaskTensorMaxLenOfficialMusa(SequenceMaskOfficialMusa):
    def setUp(self):
        self.op_type = "sequence_mask"
        self.x = np.array([[0, 3, 4], [5, 7, 9]])
        self.maxlen = 10
        self.maxlen_tensor = np.ones((1), "int32") * 10
        self.mask_dtype = "int64"
        self.inputs = {"X": self.x, "MaxLenTensor": self.maxlen_tensor}
        self.outputs = {"Y": self.calc_ground_truth_mask()}
        self.attrs = {"out_dtype": convert_np_dtype_to_proto_type(self.mask_dtype)}


class SequenceSoftmaxOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "sequence_softmax"
        self.dtype = "float64"
        x = np.random.uniform(0.1, 1, (110, 1)).astype(self.dtype)
        self.lod = [[40, 10, 30, 30]]
        out = np.zeros((110, 1)).astype(self.dtype)
        offset = 0
        for length in self.lod[0]:
            if length == 0:
                continue
            sub_x = x[offset : offset + length, :].reshape(1, length)
            out[offset : offset + length, :] = stable_softmax(sub_x).reshape(length, 1)
            offset += length

        self.inputs = {"X": (x, self.lod)}
        self.outputs = {"Out": out}
        self.attrs = {"use_cudnn": False}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_dygraph=False)


class TestSequenceAndUniformOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "sequence_mask",
            "sequence_mask_scalar",
            "sequence_softmax",
            "uniform_random_batch_size_like_sr",
            "uniform_raw_sr",
            "uniform_sr",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
