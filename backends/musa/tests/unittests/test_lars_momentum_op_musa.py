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


class TestLarsMomentumOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "lars_momentum"
        mu = 0.0001
        lars_coeff = 0.001
        lars_weight_decay = 0.0005

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        pnorm = np.sqrt(np.square(param).sum())
        gnorm = np.sqrt(np.square(grad).sum())
        local_lr = learning_rate * lars_coeff * pnorm / (
            gnorm + lars_weight_decay * param
        )
        velocity_out = mu * velocity + local_lr * (grad + lars_weight_decay * param)
        param_out = param - velocity_out

        self.inputs = {
            "Param": [("SubParam_0", param)],
            "Grad": [("SubGrad_0", grad)],
            "Velocity": [("SubVelocity_0", velocity)],
            "LearningRate": [("SubLearning_rate_0", learning_rate)],
        }
        self.attrs = {
            "mu": mu,
            "lars_coeff": lars_coeff,
            "lars_weight_decay": [lars_weight_decay],
        }
        self.outputs = {
            "ParamOut": [("SubParam_out_0", param_out)],
            "VelocityOut": [("SubVelocity_out_0", velocity_out)],
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestLarsMomentumOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("lars_momentum", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["lars_momentum"]))


if __name__ == "__main__":
    unittest.main()
