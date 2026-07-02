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
import paddle.nn.quant as Q


paddle.set_device("musa")


class TestApplyPerChannelScaleOp(unittest.TestCase):
    def _run_dynamic_case(self, rows, cols, dtype):
        np.random.seed(rows + cols)
        x_np = np.random.random((rows, cols)).astype("float32")
        scales_np = np.random.uniform(0.1, 1.0, (cols,)).astype("float32")
        x = paddle.to_tensor(x_np, dtype=dtype)
        scales = paddle.to_tensor(scales_np, dtype=dtype)

        out = Q.apply_per_channel_scale(x, scales)
        expected = (x_np * scales_np).astype("float32")
        np.testing.assert_allclose(
            out.astype("float32").numpy(), expected, rtol=1e-3, atol=1e-3
        )
        self.assertIn("musa", str(out.place))

    def test_dynamic_float16(self):
        for rows, cols in [(32, 128), (1024, 128)]:
            with self.subTest(rows=rows, cols=cols):
                self._run_dynamic_case(rows, cols, "float16")

    def test_static_float16(self):
        rows, cols = 32, 128
        np.random.seed(2026)
        x_np = np.random.random((rows, cols)).astype("float32")
        scales_np = np.random.uniform(0.1, 1.0, (cols,)).astype("float32")

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data("x", [rows, cols], dtype="float16")
            scales = paddle.static.data("scales", [cols], dtype="float16")
            out = Q.apply_per_channel_scale(x, scales)
            exe = paddle.static.Executor(paddle.CustomPlace("musa", 0))
            exe.run(startup)
            (res,) = exe.run(
                main,
                feed={
                    "x": x_np.astype("float16"),
                    "scales": scales_np.astype("float16"),
                },
                fetch_list=[out],
            )
        paddle.disable_static()

        expected = x_np * scales_np
        np.testing.assert_allclose(
            res.astype("float32"), expected, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
