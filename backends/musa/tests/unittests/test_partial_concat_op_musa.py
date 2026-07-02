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


class TestPartialConcatOpMusa(unittest.TestCase):
    def test_static_partial_concat_forward(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            x1_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="float32")
            x2_np = np.array(
                [[10, 20, 30, 40], [50, 60, 70, 80]], dtype="float32"
            )

            with paddle.static.program_guard(main, startup):
                x1 = paddle.static.data(name="x1", shape=[2, 4], dtype="float32")
                x2 = paddle.static.data(name="x2", shape=[2, 4], dtype="float32")
                out = main.current_block().create_var(
                    name="partial_concat_out", shape=[2, 4], dtype="float32"
                )
                main.current_block().append_op(
                    type="partial_concat",
                    inputs={"X": [x1, x2]},
                    outputs={"Out": out},
                    attrs={"start_index": 1, "length": 2},
                )

            exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
            exe.run(startup)
            (actual,) = exe.run(
                main,
                feed={"x1": x1_np, "x2": x2_np},
                fetch_list=[out],
            )
            expected = np.array([[2, 3, 20, 30], [6, 7, 60, 70]], dtype="float32")
            np.testing.assert_allclose(actual, expected)
            paddle.disable_static()

    def test_partial_concat_grad_kernel_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("partial_concat_grad", kernels)
        partial_concat_grad_kernels = kernels["partial_concat_grad"]
        self.assertTrue(
            any("musa" in kernel for kernel in partial_concat_grad_kernels),
            partial_concat_grad_kernels,
        )


if __name__ == "__main__":
    unittest.main()
