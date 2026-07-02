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


class TestArrayOpMusa(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("musa", 0)

    def tearDown(self):
        paddle.disable_static()

    def _run_array_to_tensor_case(self, use_stack):
        x0_np = np.arange(6, dtype="float32").reshape(2, 3)
        x1_np = (np.arange(6, dtype="float32") + 10).reshape(2, 3)
        x2_np = (np.arange(6, dtype="float32") + 20).reshape(2, 3)

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x0 = paddle.static.data("x0", [2, 3], dtype="float32")
            x1 = paddle.static.data("x1", [2, 3], dtype="float32")
            x2 = paddle.static.data("x2", [2, 3], dtype="float32")
            arr = paddle.tensor.create_array("float32")
            arr = paddle.tensor.array_write(
                x0, paddle.full([1], 0, dtype="int64"), arr
            )
            arr = paddle.tensor.array_write(
                x1, paddle.full([1], 1, dtype="int64"), arr
            )
            arr = paddle.tensor.array_write(
                x2, paddle.full([1], 2, dtype="int64"), arr
            )
            read = paddle.tensor.array_read(
                arr, paddle.full([1], 1, dtype="int64")
            )
            out, out_index = paddle.tensor.manipulation.tensor_array_to_tensor(
                arr, axis=0, use_stack=use_stack
            )

            exe = paddle.static.Executor(self.place)
            exe.run(startup)
            read_res, out_res, index_res = exe.run(
                main,
                feed={"x0": x0_np, "x1": x1_np, "x2": x2_np},
                fetch_list=[read, out, out_index],
            )

        np.testing.assert_allclose(read_res, x1_np)
        if use_stack:
            expected = np.stack([x0_np, x1_np, x2_np], axis=0)
        else:
            expected = np.concatenate([x0_np, x1_np, x2_np], axis=0)
        np.testing.assert_allclose(out_res, expected)
        np.testing.assert_array_equal(index_res, np.array([2, 2, 2], dtype="int32"))

    def test_array_to_tensor_concat(self):
        self._run_array_to_tensor_case(use_stack=False)

    def test_array_to_tensor_stack(self):
        self._run_array_to_tensor_case(use_stack=True)

    def test_tensor_to_array_grad(self):
        x_np = np.arange(6, dtype="float32").reshape(2, 3)
        y_np = (np.arange(6, dtype="float32") + 10).reshape(2, 3)

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data("x", [2, 3], dtype="float32")
            y = paddle.static.data("y", [2, 3], dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False
            arr = paddle.tensor.create_array("float32")
            arr = paddle.tensor.array_write(
                x, paddle.full([1], 0, dtype="int64"), arr
            )
            arr = paddle.tensor.array_write(
                y, paddle.full([1], 1, dtype="int64"), arr
            )
            out, _ = paddle.tensor.manipulation.tensor_array_to_tensor(
                arr, axis=0, use_stack=False
            )
            loss = paddle.sum(out)
            x_grad, y_grad = paddle.static.gradients(loss, [x, y])

            exe = paddle.static.Executor(self.place)
            exe.run(startup)
            x_grad_res, y_grad_res = exe.run(
                main,
                feed={"x": x_np, "y": y_np},
                fetch_list=[x_grad, y_grad],
            )

        np.testing.assert_allclose(x_grad_res, np.ones_like(x_np))
        np.testing.assert_allclose(y_grad_res, np.ones_like(y_np))


if __name__ == "__main__":
    unittest.main()
