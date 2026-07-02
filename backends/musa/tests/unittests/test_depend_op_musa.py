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


class TestDependOpMusa(unittest.TestCase):
    def test_static_depend_same_input_output(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            dep_np = np.array([1], dtype="int32")

            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(name="x", shape=[2, 2], dtype="float32")
                dep = paddle.static.data(name="dep", shape=[1], dtype="int32")
                main.current_block().append_op(
                    type="depend",
                    inputs={"X": x, "Dep": [dep]},
                    outputs={"Out": x},
                )
                out = x + 1.0

            exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
            exe.run(startup)
            (actual,) = exe.run(main, feed={"x": x_np, "dep": dep_np}, fetch_list=[out])
            np.testing.assert_allclose(actual, x_np + 1.0)
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
