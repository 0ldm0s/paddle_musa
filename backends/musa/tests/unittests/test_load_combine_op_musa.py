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


class TestLoadCombineOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for name in [
            "load_combine",
            "load_combine_vocab",
            "load_combine_extended",
        ]:
            self.assertIn(name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[name]))

    def test_static_save_then_load_combine_tensor(self):
        # Adapted from Paddle's save/load-combine coverage: write multiple
        # tensors into one file, then load them back sequentially on MUSA.
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()
            save_main = paddle.static.Program()
            save_startup = paddle.static.Program()
            x_np = np.full((2, 3), 2.0, dtype="float32")
            y_np = np.full((2, 2), 3.0, dtype="float32")

            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, "combined")
                with paddle.static.program_guard(save_main, save_startup):
                    x = paddle.create_parameter(
                        shape=[2, 3],
                        dtype="float32",
                        name="x",
                        default_initializer=paddle.nn.initializer.Constant(2.0),
                    )
                    y = paddle.create_parameter(
                        shape=[2, 2],
                        dtype="float32",
                        name="y",
                        default_initializer=paddle.nn.initializer.Constant(3.0),
                    )

                exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
                exe.run(save_startup)
                paddle.static.io.save_vars(
                    executor=exe,
                    dirname=tmp_dir,
                    main_program=save_main,
                    vars=[x, y],
                    filename="combined",
                )

                load_main = paddle.static.Program()
                load_startup = paddle.static.Program()
                with paddle.static.program_guard(load_main, load_startup):
                    out_x = load_main.current_block().create_var(
                        name="x", shape=[2, 3], dtype="float32"
                    )
                    out_y = load_main.current_block().create_var(
                        name="y", shape=[2, 2], dtype="float32"
                    )
                    load_main.current_block().append_op(
                        type="load_combine",
                        outputs={"Out": [out_x, out_y]},
                        attrs={
                            "file_path": file_path,
                            "load_as_fp16": False,
                            "model_from_memory": False,
                        },
                    )

                exe.run(load_startup)
                out_x_np, out_y_np = exe.run(load_main, fetch_list=[out_x, out_y])
                np.testing.assert_array_equal(out_x_np, x_np)
                np.testing.assert_array_equal(out_y_np, y_np)
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
