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


def _nadam_step(inputs, attrs):
    param = inputs["param"].copy()
    grad = inputs["grad"]
    lr = inputs["learning_rate"]
    momentum_decay_pow = inputs["momentum_decay_pow"].copy()
    beta2_pow = inputs["beta2_pow"].copy()
    mu_product = inputs["mu_product"].copy()
    moment1 = inputs["moment1"].copy()
    moment2 = inputs["moment2"].copy()

    beta1 = attrs["beta1"]
    beta2 = attrs["beta2"]
    epsilon = attrs["epsilon"]
    momentum_decay = attrs["momentum_decay"]

    momentum_decay_pow *= 0.96
    beta2_pow *= beta2
    mu_t = beta1 * (1.0 - 0.5 * (momentum_decay_pow**momentum_decay))
    mu_t_1 = beta1 * (
        1.0 - 0.5 * (momentum_decay_pow**momentum_decay) * (0.96**momentum_decay)
    )
    mu_product *= mu_t
    mu_product_t_1 = mu_product * mu_t_1
    moment1 = beta1 * moment1 + (1.0 - beta1) * grad
    moment2 = beta2 * moment2 + (1.0 - beta2) * grad * grad
    moment1_hat = mu_t_1 * moment1 / (1.0 - mu_product_t_1) + (
        1.0 - mu_t
    ) * grad / (1.0 - mu_product)
    moment2_hat = moment2 / (1.0 - beta2_pow)
    param = param - lr * moment1_hat / (np.sqrt(moment2_hat) + epsilon)
    return param, momentum_decay_pow, beta2_pow, mu_product, moment1, moment2


def _radam_step(inputs, attrs):
    param = inputs["param"].copy()
    grad = inputs["grad"]
    lr = inputs["learning_rate"]
    beta1_pow = inputs["beta1_pow"].copy()
    beta2_pow = inputs["beta2_pow"].copy()
    rho = inputs["rho"].copy()
    moment1 = inputs["moment1"].copy()
    moment2 = inputs["moment2"].copy()

    beta1 = attrs["beta1"]
    beta2 = attrs["beta2"]
    epsilon = attrs["epsilon"]
    rho_inf = 2 / (1 - beta2) - 1

    beta1_pow *= beta1
    beta2_pow *= beta2
    rho = (rho * (beta2 - beta2_pow) + beta2_pow) / (1 - beta2_pow)
    moment1 = beta1 * moment1 + (1.0 - beta1) * grad
    moment2 = beta2 * moment2 + (1.0 - beta2) * grad * grad
    moment1_hat = moment1 / (1 - beta1_pow)
    rho_t = rho_inf - 2 * rho

    if rho_t.reshape(-1)[0] > 5:
        l_t = np.sqrt(1 - beta2_pow) / (np.sqrt(moment2) + epsilon)
        r_t = np.sqrt(
            ((rho_t - 4) * (rho_t - 2) * rho_inf)
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        )
        param = param - lr * moment1_hat * r_t * l_t
    else:
        param = param - lr * moment1_hat
    return param, beta1_pow, beta2_pow, rho, moment1, moment2


class TestNAdamRAdamOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")
        np.random.seed(2026)

    def test_nadam(self):
        shape = (17, 19)
        attrs = {"beta1": 0.78, "beta2": 0.915, "epsilon": 1e-8, "momentum_decay": 0.004}
        inputs = {
            "param": np.random.uniform(-1, 1, shape).astype("float32"),
            "grad": np.random.uniform(-1, 1, shape).astype("float32"),
            "learning_rate": np.array(0.003, dtype="float32"),
            "momentum_decay_pow": (np.ones(shape) * (0.96**3)).astype("float32"),
            "beta2_pow": (np.ones(shape) * (attrs["beta2"] ** 3)).astype("float32"),
            "mu_product": (np.ones(shape) * (attrs["beta1"] ** 3)).astype("float32"),
            "moment1": np.random.uniform(-1, 1, shape).astype("float32"),
            "moment2": np.random.random(shape).astype("float32"),
        }
        expected = _nadam_step(inputs, attrs)
        outs = paddle._C_ops.nadam_(
            paddle.to_tensor(inputs["param"]),
            paddle.to_tensor(inputs["grad"]),
            paddle.to_tensor(inputs["learning_rate"]),
            paddle.to_tensor(inputs["momentum_decay_pow"]),
            paddle.to_tensor(inputs["beta2_pow"]),
            paddle.to_tensor(inputs["mu_product"]),
            paddle.to_tensor(inputs["moment1"]),
            paddle.to_tensor(inputs["moment2"]),
            None,
            attrs["beta1"],
            attrs["beta2"],
            attrs["epsilon"],
            attrs["momentum_decay"],
            False,
        )
        for actual, expect in zip(outs[:6], expected):
            np.testing.assert_allclose(actual.numpy(), expect, rtol=1e-6, atol=1e-6)

    def test_radam(self):
        shape = (17, 19)
        attrs = {"beta1": 0.78, "beta2": 0.915, "epsilon": 1e-8}
        rho_inf = 2 / (1 - attrs["beta2"]) - 1
        inputs = {
            "param": np.random.uniform(-1, 1, shape).astype("float32"),
            "grad": np.random.uniform(-1, 1, shape).astype("float32"),
            "learning_rate": np.array(0.003, dtype="float32"),
            "beta1_pow": (np.ones(shape) * (attrs["beta1"] ** 3)).astype("float32"),
            "beta2_pow": (np.ones(shape) * (attrs["beta2"] ** 3)).astype("float32"),
            "rho": (np.ones(shape) * ((rho_inf - 5) / 2 + 5.0)).astype("float32"),
            "moment1": np.random.uniform(-1, 1, shape).astype("float32"),
            "moment2": np.random.random(shape).astype("float32"),
        }
        expected = _radam_step(inputs, attrs)
        outs = paddle._C_ops.radam_(
            paddle.to_tensor(inputs["param"]),
            paddle.to_tensor(inputs["grad"]),
            paddle.to_tensor(inputs["learning_rate"]),
            paddle.to_tensor(inputs["beta1_pow"]),
            paddle.to_tensor(inputs["beta2_pow"]),
            paddle.to_tensor(inputs["rho"]),
            paddle.to_tensor(inputs["moment1"]),
            paddle.to_tensor(inputs["moment2"]),
            None,
            attrs["beta1"],
            attrs["beta2"],
            attrs["epsilon"],
            False,
        )
        for actual, expect in zip(outs[:6], expected):
            np.testing.assert_allclose(actual.numpy(), expect, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
