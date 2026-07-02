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

import paddle


class TestMultinomialOpMusa(unittest.TestCase):
    def test_multinomial_replacement(self):
        paddle.set_device("musa")
        probs = paddle.to_tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype="float32")
        out = paddle.multinomial(probs, num_samples=2, replacement=True)
        self.assertEqual(list(out.shape), [2, 2])
        self.assertEqual(out.dtype, paddle.int64)
        self.assertTrue(bool(((out >= 0) & (out < 3)).all().item()))


if __name__ == "__main__":
    unittest.main()
