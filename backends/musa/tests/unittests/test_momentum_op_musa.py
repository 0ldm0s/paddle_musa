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
from op import Operator
from op_test import OpTest
from paddle.base import core


def get_places(self):
    return [paddle.CustomPlace("musa", 0)]


OpTest._get_places = get_places


def calculate_momentum_by_numpy(
    param,
    grad,
    mu,
    velocity,
    use_nesterov,
    learning_rate,
    regularization_method=None,
    regularization_coeff=1.0,
):
    if regularization_method == "l2_decay":
        grad = grad + regularization_coeff * param

    velocity_out = mu * velocity + grad
    if use_nesterov:
        param_out = param - (grad + velocity_out * mu) * learning_rate
    else:
        param_out = param - learning_rate * velocity_out
    return param_out, velocity_out


class TestMomentumOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "momentum"
        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = False
        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Velocity": velocity,
            "LearningRate": learning_rate,
        }
        self.attrs = {"mu": mu}
        param_out, velocity_out = calculate_momentum_by_numpy(
            param, grad, mu, velocity, use_nesterov, learning_rate
        )
        self.outputs = {"ParamOut": param_out, "VelocityOut": velocity_out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestMomentumNesterovOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "momentum"
        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        velocity = np.zeros((123, 321)).astype("float32")
        learning_rate = np.array([0.001]).astype("float32")
        mu = 0.0001
        use_nesterov = True
        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Velocity": velocity,
            "LearningRate": learning_rate,
        }
        self.attrs = {"mu": mu, "use_nesterov": use_nesterov}
        param_out, velocity_out = calculate_momentum_by_numpy(
            param, grad, mu, velocity, use_nesterov, learning_rate
        )
        self.outputs = {"ParamOut": param_out, "VelocityOut": velocity_out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestSparseMomentumOfficialMusa(unittest.TestCase):
    def check_with_place(self, place, use_nesterov=False):
        scope = core.Scope()
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        mu = 1.0

        param = scope.var("Param").get_tensor()
        param_array = np.full((height, row_numel), 5.0).astype("float32")
        param.set(param_array, place)
        param_out = scope.var("ParamOut").get_tensor()
        param_out.set(np.full((height, row_numel), 0.0).astype("float32"), place)

        grad_selected_rows = scope.var("Grad").get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        grad_np_array = np.ones((len(rows), row_numel)).astype("float32")
        grad_np_array[0, 0] = 2.0
        grad_np_array[2, 8] = 4.0
        grad_selected_rows.get_tensor().set(grad_np_array, place)

        velocity = scope.var("Velocity").get_tensor()
        velocity_np_array = np.ones((height, row_numel)).astype("float32")
        velocity.set(velocity_np_array, place)
        velocity_out = scope.var("VelocityOut").get_tensor()
        velocity_out.set(np.full((height, row_numel), 0.0).astype("float32"), place)

        lr = scope.var("LearningRate").get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        op = Operator(
            "momentum",
            Param="Param",
            Grad="Grad",
            Velocity="Velocity",
            ParamOut="ParamOut",
            VelocityOut="VelocityOut",
            LearningRate="LearningRate",
            mu=mu,
            use_nesterov=use_nesterov,
        )
        op.run(scope, place)

        dense_grad = np.full((height, row_numel), 0.0).astype("float32")
        for i, row in enumerate(rows):
            dense_grad[row] = grad_np_array[i]
        expected_param, expected_velocity = calculate_momentum_by_numpy(
            param_array, dense_grad, mu, velocity_np_array, use_nesterov, lr_array
        )
        np.testing.assert_allclose(np.array(param_out), expected_param)
        np.testing.assert_allclose(np.array(velocity_out), expected_velocity)

    def test_sparse_momentum(self):
        self.check_with_place(paddle.CustomPlace("musa", 0), use_nesterov=False)

    def test_sparse_momentum_nesterov(self):
        self.check_with_place(paddle.CustomPlace("musa", 0), use_nesterov=True)



class TestMomentumOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["momentum", "momentum_dense_param_sparse_grad"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
