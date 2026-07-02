import unittest

import paddle
from paddle.base import core


class TestSparseGradExtraOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "add_coo_coo_grad",
            "add_coo_dense_grad",
            "cast_coo_grad",
            "cast_csr_grad",
            "coo_to_dense_grad",
            "mask_as_coo_grad",
            "mask_as_csr_grad",
            "masked_matmul_csr_grad",
            "matmul_coo_coo_grad",
            "matmul_coo_dense_grad",
            "matmul_csr_csr_grad",
            "matmul_csr_dense_grad",
            "mv_coo_grad",
            "mv_csr_grad",
            "sparse_coo_tensor_grad",
            "values_coo_grad",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
