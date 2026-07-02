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

import os
import tempfile
import unittest

import numpy as np
import paddle
from paddle.base import core


class TestLoadOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("load", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["load"]))

    def test_static_save_then_load_tensor(self):
        # Mirrors Paddle's save/load coverage by writing a tensor with the save
        # operator and reading it back with the load operator on the target place.
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()
            main = paddle.static.Program()
            startup = paddle.static.Program()
            x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, "tensor")
                with paddle.static.program_guard(main, startup):
                    x = paddle.static.data(name="x", shape=[2, 2], dtype="float32")
                    out = main.current_block().create_var(
                        name="out", shape=[2, 2], dtype="float32"
                    )
                    main.current_block().append_op(
                        type="save",
                        inputs={"X": x},
                        attrs={
                            "file_path": file_path,
                            "overwrite": True,
                            "save_as_fp16": False,
                        },
                    )
                    main.current_block().append_op(
                        type="load",
                        outputs={"Out": out},
                        attrs={
                            "file_path": file_path,
                            "seek": -1,
                            "shape": [],
                            "load_as_fp16": False,
                        },
                    )

                exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
                exe.run(startup)
                (out_np,) = exe.run(main, feed={"x": x_np}, fetch_list=[out])
                np.testing.assert_array_equal(out_np, x_np)
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
