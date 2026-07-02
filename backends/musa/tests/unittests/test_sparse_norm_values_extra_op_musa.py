import copy
import unittest

import numpy as np

import paddle
from paddle.base import core


class TestSparseNormValuesExtraOpMusa(unittest.TestCase):
    def setUp(self):
        paddle.set_device("musa")

    def test_registered(self):
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "batch_norm_coo",
            "batch_norm_coo_grad",
            "skip_layernorm",
            "sparse_coo_tensor",
            "sparse_momentum",
            "sync_batch_norm_coo",
            "sync_batch_norm_coo_grad",
            "values_coo",
            "values_csr",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))

    def _check_sparse_batch_norm(self, shape):
        paddle.seed(0)
        dense_x = paddle.randn(shape)
        dense_x.stop_gradient = False

        data_format = "NHWC" if len(shape) == 4 else "NDHWC"
        if len(shape) == 4:
            batch_norm = paddle.nn.BatchNorm2D(shape[-1], data_format=data_format)
        else:
            batch_norm = paddle.nn.BatchNorm3D(shape[-1], data_format=data_format)
        dense_y = batch_norm(dense_x)
        dense_y.backward(dense_y)

        sparse_dim = len(shape) - 1
        sparse_dense_x = copy.deepcopy(dense_x)
        sparse_dense_x.stop_gradient = False
        sparse_x = sparse_dense_x.to_sparse_coo(sparse_dim)
        sparse_x.retain_grads()
        sparse_batch_norm = paddle.sparse.nn.BatchNorm(
            shape[-1], data_format=data_format
        )
        sparse_batch_norm._mean.set_value(batch_norm._mean)
        sparse_batch_norm._variance.set_value(batch_norm._variance)
        sparse_batch_norm.weight.set_value(batch_norm.weight)

        sparse_y = sparse_batch_norm(sparse_x)
        np.testing.assert_allclose(
            dense_y.flatten().numpy(),
            sparse_y.values().flatten().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

        sparse_y.backward(sparse_y)
        np.testing.assert_allclose(
            dense_x.grad.flatten().numpy(),
            sparse_x.grad.values().flatten().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_sparse_batch_norm_official_cases(self):
        self._check_sparse_batch_norm([2, 4, 4, 3])
        self._check_sparse_batch_norm([2, 4, 4, 3, 2])

    def test_sparse_batch_norm_static_official_case(self):
        paddle.enable_static()
        main_program = paddle.base.Program()
        startup_program = paddle.base.Program()
        with paddle.base.program_guard(main_program, startup_program):
            indices = paddle.static.data(
                name="indices", shape=[4, 4], dtype="int32"
            )
            values = paddle.static.data(
                name="values", shape=[4, 1], dtype="float32"
            )
            channels = 1
            dense_shape = [1, 1, 3, 4, channels]
            sp_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
            sp_y = paddle.sparse.nn.BatchNorm(channels)(sp_x)
            out = sp_y.to_dense()

            exe = paddle.static.Executor()
            exe.run(startup_program)
            indices_data = [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [1, 3, 2, 3],
            ]
            values_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype(
                "float32"
            )
            fetch = exe.run(
                main_program,
                feed={"indices": indices_data, "values": values_data},
                fetch_list=[out],
                return_numpy=True,
            )
            correct_out = np.array(
                [
                    [
                        [
                            [[0.0], [-1.3416353], [0.0], [-0.44721174]],
                            [[0.0], [0.0], [0.44721198], [0.0]],
                            [[0.0], [0.0], [0.0], [1.3416355]],
                        ]
                    ]
                ]
            ).astype("float32")
            np.testing.assert_allclose(correct_out, fetch[0], rtol=1e-5)
        paddle.disable_static()

    def test_sync_batch_norm_official_case(self):
        x = np.array(
            [[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]
        ).astype("float32")
        dense_x = paddle.to_tensor(x)
        sparse_x = dense_x.to_sparse_coo(len(x.shape) - 1)

        sparse_sync_bn = paddle.sparse.nn.SyncBatchNorm(2)
        sparse_hidden = sparse_sync_bn(sparse_x)

        dense_sync_bn = paddle.nn.SyncBatchNorm(2)
        dense_hidden = dense_sync_bn(dense_x.reshape((-1, dense_x.shape[-1])))
        np.testing.assert_allclose(
            sparse_hidden.values().numpy(), dense_hidden.numpy(), rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
