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

import sys
import unittest
from pathlib import Path

import numpy as np
import paddle

sys.path.append(str(Path(__file__).resolve().parents[4] / "Paddle" / "test" / "legacy_test"))
from op import Operator
from paddle.base import core


class TestMergeSelectedRowsOfficialMusa(unittest.TestCase):
    def test_check_output(self):
        place = paddle.CustomPlace("musa", 0)
        scope = core.Scope()
        x_rows = [0, 5, 5, 4, 19]
        out_rows = [0, 4, 5, 19]
        height = 20
        row_numel = 2

        np_array = np.ones((len(x_rows), row_numel)).astype("float32")
        np_array[1, :] = 2.0
        np_array[2, :] = 3.0
        np_array[3, :] = 4.0

        x = scope.var("X").get_selected_rows()
        x.set_rows(x_rows)
        x.set_height(height)
        x.get_tensor().set(np_array, place)
        out = scope.var("Out").get_selected_rows()

        op = Operator("merge_selected_rows", X="X", Out="Out")
        op.run(scope, place)

        self.assertEqual(out.rows(), out_rows)
        self.assertEqual(out.height(), height)
        out_array = np.array(out.get_tensor())
        self.assertEqual((4, 2), out_array.shape)
        np.testing.assert_array_equal(out_array[0, :], np.array([1.0, 1.0]))
        np.testing.assert_array_equal(out_array[1, :], np.array([4.0, 4.0]))
        np.testing.assert_array_equal(out_array[2, :], np.array([5.0, 5.0]))
        np.testing.assert_array_equal(out_array[3, :], np.array([1.0, 1.0]))


class TestLookupTableSelectedRowsOfficialMusa(unittest.TestCase):
    def test_w_is_selected_rows(self):
        place = paddle.CustomPlace("musa", 0)
        scope = core.Scope()
        ids_array = np.array([[0], [4], [3], [5]]).astype("int64")
        scope.var("Ids").get_tensor().set(ids_array, place)

        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12
        w_selected_rows = scope.var("W").get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype("float32")
        for i in range(len(rows)):
            w_array[i] *= i
        w_selected_rows.get_tensor().set(w_array, place)

        out_tensor = scope.var("Out").get_tensor()
        lookup_table = Operator("lookup_table", W="W", Ids="Ids", Out="Out")
        lookup_table.run(scope, place)
        result_array = np.array(out_tensor)

        for idx, row in enumerate(ids_array.flatten()):
            np.testing.assert_array_equal(result_array[idx], np.full(row_numel, row))


class TestSelectedRowsBasicOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "add_n_sr",
            "clip_sr",
            "ftrl_sr",
            "full_sr",
            "full_with_tensor_sr",
            "isfinite_sr",
            "isinf_sr",
            "isnan_sr",
            "lamb_sr",
            "load_sr",
            "lookup_table_sr",
            "merge_selected_rows",
            "multiply_raw_sr",
            "multiply_sr",
            "save_sr",
            "scale_sr",
            "shape64_sr",
            "shape_sr",
            "share_data_sr",
            "square_sr",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
