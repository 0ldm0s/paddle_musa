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


class TestGammainccOpMusa(unittest.TestCase):
    def test_gammaincc(self):
        paddle.set_device("musa")
        x = paddle.to_tensor(np.array([1.0, 2.0, 3.0], dtype="float32"))
        y = paddle.to_tensor(np.array([0.5, 1.0, 2.0], dtype="float32"))
        out = paddle.gammaincc(x, y)
        self.assertEqual(out.shape, [3])
        self.assertTrue(np.all(np.isfinite(out.numpy())))


if __name__ == "__main__":
    unittest.main()
