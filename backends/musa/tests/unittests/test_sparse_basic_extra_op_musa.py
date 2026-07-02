import unittest

import paddle
from paddle.base import core


class TestSparseBasicExtraOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "add_coo_coo",
            "add_coo_dense",
            "cast_coo",
            "cast_csr",
            "coalesce_coo",
            "coo_to_csr",
            "coo_to_dense",
            "csr_to_coo",
            "csr_to_dense",
            "dense_to_coo",
            "dense_to_csr",
            "divide_scalar_coo",
            "divide_scalar_csr",
            "empty_like_coo",
            "empty_like_csr",
            "full_like_coo",
            "full_like_csr",
            "indices_coo",
            "isnan_coo",
            "isnan_csr",
            "mask_as_coo",
            "mask_as_csr",
            "mask_helper_coo",
            "masked_matmul_csr",
            "matmul_coo_coo",
            "matmul_coo_dense",
            "matmul_csr_csr",
            "matmul_csr_dense",
            "mv_coo",
            "mv_csr",
            "slice_coo",
            "slice_csr",
            "transpose_coo",
            "transpose_csr",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
