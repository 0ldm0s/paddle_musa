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

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import paddle

sys.path.append(str(Path(__file__).resolve().parents[4] / "Paddle" / "test" / "legacy_test"))
from op_test import OpTest
from paddle.base import core


def get_places(self):
    return [paddle.CustomPlace("musa", 0)]


OpTest._get_places = get_places


def quantize_max_abs(x, max_range):
    scale = np.max(np.abs(x).flatten())
    y = np.round(x / scale * max_range)
    return y, scale


def dequantize_max_abs(x, scale, max_range):
    return x * scale / max_range


def channel_wise_quantize_max_abs(x, quant_bit=8, quant_axis=0):
    assert quant_axis in [0, 1], "The quant_axis should be 0 or 1."
    scales = []
    y = x.copy()
    max_range = math.pow(2, quant_bit - 1) - 1
    if quant_axis == 0:
        for i in range(x.shape[0]):
            scale = np.max(np.abs(x[i])).astype("float32")
            scales.append(scale)
            y[i] = np.round(x[i] * max_range / scale)
    elif quant_axis == 1:
        for i in range(x.shape[1]):
            scale = np.max(np.abs(x[:, i])).astype("float32")
            scales.append(scale)
            y[:, i] = np.round(x[:, i] * max_range / scale)
    return y, scales


def channel_wise_dequantize_max_abs(x, scales, quant_bits, quant_axis):
    assert quant_axis in [0, 1], "The quant_axis should be 0 or 1."
    max_range = math.pow(2, quant_bits - 1) - 1
    y = x.copy()
    if quant_axis == 0:
        for i in range(x.shape[0]):
            y[i] = x[i] * scales[i] / max_range
    elif quant_axis == 1:
        for i in range(x.shape[1]):
            y[:, i] = x[:, i] * scales[i] / max_range
    return y


class TestQuantizeLinearOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "quantize_linear"
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        x = np.random.randn(31, 65).astype("float32")
        yq, scale = quantize_max_abs(x, self.max_range)
        scale = np.array(scale).astype("float32")
        zero_point = np.zeros(scale.shape, dtype="int32")
        self.inputs = {"X": x, "Scale": scale, "ZeroPoint": zero_point}
        self.attrs = {"bit_length": self.bit_length, "quant_axis": self.quant_axis}
        self.outputs = {"Y": yq}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestQuantizeLinearTrainOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "quantize_linear"
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.attrs = {
            "bit_length": self.bit_length,
            "quant_axis": self.quant_axis,
            "moving_rate": 0.9,
            "is_test": False,
        }
        x = np.random.randn(31, 65).astype("float32")
        scale = np.array([0.001]).astype("float32")
        zero_point = np.zeros(scale.shape, dtype="int32")
        in_accum = np.ones(1).astype("float32")
        in_state = np.ones(1).astype("float32")
        out_accum = self.attrs["moving_rate"] * in_accum + np.max(np.abs(x))
        out_state = self.attrs["moving_rate"] * in_state + 1.0
        out_scale = out_accum / out_state
        round_out = np.round(x / out_scale * self.max_range)
        quant_data = np.clip(round_out, -self.max_range - 1, self.max_range)
        self.inputs = {
            "X": x,
            "Scale": scale,
            "ZeroPoint": zero_point,
            "InAccum": in_accum,
            "InState": in_state,
        }
        self.outputs = {
            "Y": quant_data,
            "OutScale": out_scale,
            "OutAccum": out_accum,
            "OutState": out_state,
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestChannelWiseQuantizeLinearOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "quantize_linear"
        self.bit_length = 8
        self.quant_axis = 0
        x = np.random.randn(4, 3, 64, 64).astype("float32")
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length, self.quant_axis)
        scale = np.array(scale).astype("float32")
        zero_point = np.zeros(scale.shape, dtype="int32")
        self.inputs = {"X": x, "Scale": scale, "ZeroPoint": zero_point}
        self.attrs = {"bit_length": self.bit_length, "quant_axis": self.quant_axis}
        self.outputs = {"Y": yq}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestDequantizeLinearOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "dequantize_linear"
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        x = np.random.randn(31, 65).astype("float32")
        yq, scale = quantize_max_abs(x, self.max_range)
        ydq = dequantize_max_abs(yq, scale, self.max_range)
        scale = np.array(scale).astype("float32")
        zero_point = np.zeros(scale.shape, dtype="int32")
        self.inputs = {"X": yq, "Scale": scale, "ZeroPoint": zero_point}
        self.attrs = {
            "bit_length": self.bit_length,
            "quant_axis": self.quant_axis,
            "qmin": int(-1 * self.max_range - 1),
            "qmax": int(self.max_range),
        }
        self.outputs = {"Y": ydq}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestChannelWiseDequantizeLinearOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "dequantize_linear"
        self.bit_length = 8
        self.quant_axis = 0
        x = np.random.randn(4, 3, 64, 64).astype("float32")
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length, self.quant_axis)
        ydq = channel_wise_dequantize_max_abs(
            yq, scale, self.bit_length, self.quant_axis
        )
        scale = np.array(scale).astype("float32")
        zero_point = np.zeros(scale.shape, dtype="int32")
        self.inputs = {"X": yq, "Scale": scale, "ZeroPoint": zero_point}
        self.attrs = {"bit_length": self.bit_length, "quant_axis": self.quant_axis}
        self.outputs = {"Y": ydq}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestDequantizeLinearOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "dequantize_linear",
            "dequantize_linear_deprecated",
            "quantize_linear",
            "quantize_linear_deprecated_train",
            "quantize_linear_deprecated_infer",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
