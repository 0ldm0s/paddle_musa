# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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


class TestAccuracyCheckOp(unittest.TestCase):
    def _run_equal_case(self, dtype):
        if dtype == "bfloat16":
            x = paddle.to_tensor(np.array([1, 2, 3, 4], dtype="float32")).astype(
                "bfloat16"
            )
            y = x.clone()
        else:
            x_np = np.array([1, 2, 3, 4], dtype=dtype)
            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(x_np.copy())

        out = paddle._C_ops.accuracy_check(
            x, y, "accuracy_check_" + dtype, 1e-5, 1e-8, False
        )
        self.assertIn("musa", str(out.place))
        np.testing.assert_array_equal(
            out.numpy(), np.ones([4], dtype=np.bool_)
        )

    def test_equal_inputs(self):
        for dtype in (
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "int32",
            "int64",
            "uint8",
            "int8",
            "int16",
            "bool",
            "complex64",
            "complex128",
        ):
            with self.subTest(dtype=dtype):
                self._run_equal_case(dtype)

    def test_equal_nan(self):
        x = paddle.to_tensor(np.array([1.0, np.nan], dtype="float32"))
        y = paddle.to_tensor(np.array([1.0, np.nan], dtype="float32"))
        out = paddle._C_ops.accuracy_check(
            x, y, "accuracy_check_equal_nan", 1e-5, 1e-8, True
        )
        np.testing.assert_array_equal(
            out.numpy(), np.ones([2], dtype=np.bool_)
        )

    def test_raises_on_mismatch(self):
        x = paddle.to_tensor(np.array([1.0, 2.0], dtype="float32"))
        y = paddle.to_tensor(np.array([1.0, 3.0], dtype="float32"))
        with self.assertRaisesRegex(RuntimeError, "Accuracy check failed"):
            paddle._C_ops.accuracy_check(
                x, y, "accuracy_check_mismatch", 1e-5, 1e-8, False
            )


if __name__ == "__main__":
    unittest.main()
