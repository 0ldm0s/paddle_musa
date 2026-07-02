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


paddle.set_device("musa")


class TestASGDOpMusa(unittest.TestCase):
    def _run_case(self, dtype):
        np.random.seed(2026)
        param_np = np.random.random((32, 64)).astype("float32")
        grad_np = np.random.random((32, 64)).astype("float32")
        d_np = np.random.random((32, 64)).astype("float32")
        y_np = np.random.random((32, 64)).astype("float32")
        lr_np = np.array([0.001], dtype="float32")
        n_np = np.array([1000], dtype="float32")

        param = paddle.to_tensor(param_np, dtype=dtype)
        grad = paddle.to_tensor(grad_np, dtype=dtype)
        d = paddle.to_tensor(d_np, dtype=dtype)
        y = paddle.to_tensor(y_np, dtype=dtype)
        lr = paddle.to_tensor(lr_np, dtype=dtype)
        n = paddle.to_tensor(n_np, dtype=dtype)

        paddle._C_ops.asgd_(param, grad, lr, d, y, n, None, False)

        d_out = d_np - y_np + grad_np
        y_out = grad_np
        param_out = param_np - (lr_np / n_np) * d_out

        atol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-6
        rtol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-6
        np.testing.assert_allclose(
            param.astype("float32").numpy(), param_out, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            d.astype("float32").numpy(), d_out, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            y.astype("float32").numpy(), y_out, rtol=rtol, atol=atol
        )
        self.assertIn("musa", str(param.place))

    def test_float32(self):
        self._run_case("float32")

    def test_float16(self):
        self._run_case("float16")

    def test_bfloat16(self):
        self._run_case("bfloat16")

    def test_optimizer_smoke(self):
        paddle.seed(2026)
        x = paddle.randn((4, 8), dtype="float32")
        linear = paddle.nn.Linear(8, 3)
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.001, batch_num=2, parameters=linear.parameters()
        )
        out = linear(x)
        loss = paddle.mean(out)
        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()


if __name__ == "__main__":
    unittest.main()
