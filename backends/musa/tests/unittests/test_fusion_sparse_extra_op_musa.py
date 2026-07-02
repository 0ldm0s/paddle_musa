import unittest

import paddle
from paddle.base import core


class TestFusionSparseExtraOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "fc",
            "gemm_epilogue",
            "maxpool_coo",
            "multihead_matmul",
            "resnet_unit",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
