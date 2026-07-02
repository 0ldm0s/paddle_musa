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


class TestMultiDotOpMusa(unittest.TestCase):
    def test_multi_dot(self):
        paddle.set_device("musa")
        x = paddle.to_tensor(np.arange(6, dtype="float32").reshape(2, 3))
        y = paddle.to_tensor(np.arange(12, dtype="float32").reshape(3, 4))
        z = paddle.to_tensor(np.arange(8, dtype="float32").reshape(4, 2))
        out = paddle.linalg.multi_dot([x, y, z])
        expected = np.linalg.multi_dot([x.numpy(), y.numpy(), z.numpy()])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
