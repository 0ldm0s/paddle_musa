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

import paddle
from paddle.base import core


class TestSaveCombineOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in ["save_combine_tensor", "save_combine_vocab"]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))

    def test_static_save_combine_tensor(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()
            main = paddle.static.Program()
            startup = paddle.static.Program()

            with paddle.static.program_guard(main, startup):
                w = paddle.create_parameter(
                    shape=[2, 2],
                    dtype="float32",
                    name="save_combine_w",
                    default_initializer=paddle.nn.initializer.Constant(2.0),
                )
                b = paddle.create_parameter(
                    shape=[2],
                    dtype="float32",
                    name="save_combine_b",
                    default_initializer=paddle.nn.initializer.Constant(1.0),
                )

            exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
            exe.run(startup)
            with tempfile.TemporaryDirectory() as tmp_dir:
                paddle.static.io.save_vars(
                    executor=exe,
                    dirname=tmp_dir,
                    main_program=main,
                    vars=[w, b],
                    filename="vars_file",
                )
                file_path = os.path.join(tmp_dir, "vars_file")
                self.assertTrue(os.path.exists(file_path))
                self.assertGreater(os.path.getsize(file_path), 0)
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
