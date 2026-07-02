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


class TestMovingAverageAbsMaxScaleOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "moving_average_abs_max_scale"
        self.attrs = {"moving_rate": 0.9, "is_test": False}
        input_data = np.random.random((8, 16, 7, 7)).astype("float32")
        in_accum = np.ones(1).astype("float32")
        in_state = np.ones(1).astype("float32")
        out_accum = self.attrs["moving_rate"] * in_accum + np.max(np.abs(input_data))
        out_state = self.attrs["moving_rate"] * in_state + 1.0
        out_scale = out_accum / out_state
        self.inputs = {"X": input_data, "InAccum": in_accum, "InState": in_state}
        self.outputs = {
            "Out": input_data,
            "OutAccum": out_accum,
            "OutState": out_state,
            "OutScale": out_scale,
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)



class TestMovingAverageAbsMaxScaleOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        name = "moving_average_abs_max_scale"
        self.assertIn(name, kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels[name]))


if __name__ == "__main__":
    unittest.main()
