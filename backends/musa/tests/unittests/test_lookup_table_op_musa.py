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
from op_test import OpTest
from paddle.base import core


def get_places(self):
    return [paddle.CustomPlace("musa", 0)]


OpTest._get_places = get_places


class TestLookupTableOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(0, 17, (4, 1)).astype("int64")
        flat_ids = ids.flatten()
        self.inputs = {"W": table, "Ids": ids}
        self.outputs = {"Out": table[flat_ids].reshape((4, 31))}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(["W"], "Out", no_grad_set=set("Ids"), check_dygraph=False)


class TestLookupTableTensorIdsOfficialMusa(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float64")
        ids = np.random.randint(0, 17, (2, 4, 5, 1)).astype("int64")
        self.inputs = {"W": table, "Ids": ids}
        self.outputs = {"Out": table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(["W"], "Out", no_grad_set=set("Ids"), check_dygraph=False)



class TestLookupTableOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        self.assertIn("lookup_table", kernels)
        self.assertTrue(any("musa" in kernel for kernel in kernels["lookup_table"]))


if __name__ == "__main__":
    unittest.main()
