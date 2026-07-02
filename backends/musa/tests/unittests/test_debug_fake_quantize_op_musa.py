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


def get_places(self):
    return [paddle.CustomPlace("musa", 0)]


OpTest._get_places = get_places


def round_c_single_element(val):
    dtype = type(val)
    if val >= 0:
        return dtype(np.floor(val + 0.5))
    return dtype(np.ceil(val - 0.5))


round_c = np.vectorize(round_c_single_element)


def get_compute_type(dtype):
    if dtype == np.float16:
        return np.float32
    return dtype


class TestFakeQuantizeAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_abs_max"
        self.attrs = {"bit_length": 8, "round_type": 1}
        input_data = np.random.random((12, 24)).astype("float32")
        scale = np.max(np.abs(input_data)).flatten()
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        output_data = round_c(input_data * bnt / scale)
        self.inputs = {"X": input_data}
        self.outputs = {"Out": output_data, "OutScale": scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFakeChannelWiseQuantizeAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_channel_wise_quantize_abs_max"
        self.attrs = {"bit_length": 8, "round_type": 0, "quant_axis": 1}
        input_data = np.random.random((6, 4, 5, 5)).astype("float32")
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        scale_broadcast = np.amax(input_data, axis=(0, 2, 3), keepdims=True)
        output_data = np.clip(np.round(input_data / scale_broadcast * bnt), -bnt - 1, bnt)
        scale = np.transpose(scale_broadcast, (1, 0, 2, 3)).reshape(input_data.shape[1], -1)[:, 0]
        self.inputs = {"X": input_data}
        self.outputs = {"Out": output_data, "OutScale": scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFakeChannelWiseQuantizeDequantizeAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_channel_wise_quantize_dequantize_abs_max"
        self.attrs = {"bit_length": 8, "round_type": 0, "quant_axis": 1}
        input_data = np.random.random((6, 4, 5, 5)).astype("float32")
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        scale_broadcast = np.amax(input_data, axis=(0, 2, 3), keepdims=True)
        round_out = np.round(input_data / scale_broadcast * bnt)
        output_data = np.clip(round_out, -bnt - 1, bnt) * scale_broadcast / bnt
        scale = np.transpose(scale_broadcast, (1, 0, 2, 3)).reshape(input_data.shape[1], -1)[:, 0]
        self.inputs = {"X": input_data}
        self.outputs = {"Out": output_data, "OutScale": scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        gradient = [np.ones(self.inputs["X"].shape) / np.prod(self.inputs["X"].shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient, check_dygraph=False)


class TestFakeQuantizeRangeAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_range_abs_max"
        self.attrs = {"bit_length": 5, "window_size": 1, "is_test": False, "round_type": 0}
        input_data = ((np.random.random((4, 8, 3, 3)) - 0.4) * 10).astype("float32")
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        in_scale = np.zeros(1).astype("float32")
        out_scale = np.array([np.max(np.abs(input_data))]).astype("float32")
        output_data = np.clip(np.round(input_data / out_scale[0] * bnt), -bnt - 1, bnt)
        self.inputs = {"X": input_data, "Iter": np.zeros(1).astype("int64"), "InScale": in_scale}
        self.outputs = {"Out": output_data, "OutScale": out_scale, "OutScales": out_scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFakeQuantizeMovingAverageAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_moving_average_abs_max"
        self.attrs = {"bit_length": 5, "moving_rate": 0.9, "is_test": False, "round_type": 1}
        input_data = np.random.random((4, 8, 3, 3)).astype("float32")
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        in_scale = np.array([0.001]).astype("float32")
        in_accum = np.ones(1).astype("float32")
        in_state = np.ones(1).astype("float32")
        out_accum = self.attrs["moving_rate"] * in_accum + np.max(np.abs(input_data))
        out_state = self.attrs["moving_rate"] * in_state + 1.0
        out_scale = out_accum / out_state
        output_data = round_c(input_data / out_scale * bnt).astype("float32")
        self.inputs = {"X": input_data, "InScale": in_scale, "InAccum": in_accum, "InState": in_state}
        self.outputs = {"Out": output_data, "OutAccum": out_accum, "OutState": out_state, "OutScale": out_scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFakeQuantizeDequantizeMovingAverageAbsMaxMusa(TestFakeQuantizeMovingAverageAbsMaxMusa):
    def setUp(self):
        super().setUp()
        self.op_type = "fake_quantize_dequantize_moving_average_abs_max"
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        self.outputs["Out"] = self.outputs["Out"] * self.outputs["OutScale"] / bnt

    def test_check_grad(self):
        gradient = [np.ones(self.inputs["X"].shape) / np.prod(self.inputs["X"].shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient, check_dygraph=False)


class TestFakeQuantizeDequantizeAbsMaxMusa(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_dequantize_abs_max"
        self.attrs = {"bit_length": 8, "round_type": 1}
        input_data = np.random.random((12, 24)).astype("float32")
        scale = np.max(np.abs(input_data)).flatten().astype("float32")
        bnt = (1 << (self.attrs["bit_length"] - 1)) - 1
        output_data = round_c(input_data / scale * bnt) * scale / bnt
        self.inputs = {"X": input_data}
        self.outputs = {"Out": output_data, "OutScale": scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        gradient = [np.ones(self.inputs["X"].shape) / np.prod(self.inputs["X"].shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient, check_dygraph=False)


class TestDebugAndFakeQuantizeRegistryMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "check_model_nan_inf",
            "fake_quantize_range_abs_max",
            "fake_quantize_abs_max",
            "fake_quantize_moving_average_abs_max",
            "fake_channel_wise_quantize_abs_max",
            "fake_channel_wise_quantize_dequantize_abs_max",
            "fake_quantize_dequantize_moving_average_abs_max",
            "fake_quantize_dequantize_abs_max",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]), kernels[op_name])


if __name__ == "__main__":
    unittest.main()
