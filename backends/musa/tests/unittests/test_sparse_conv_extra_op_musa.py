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


def seqconv(
    x,
    lod,
    filter,
    context_length,
    context_start,
    padding_trainable=False,
    padding_data=None,
):
    [time_steps, channels] = x.shape
    col = np.zeros((time_steps, context_length * channels)).astype("float32")
    offset = [0]
    for seq_len in lod[0]:
        offset.append(offset[-1] + seq_len)
    begin_pad = np.max([0, -context_start])
    for i in range(len(offset) - 1):
        for j in range(context_length):
            in_begin = offset[i] + context_start + j
            in_end = offset[i + 1] + context_start + j
            out_begin = offset[i]
            out_end = offset[i + 1]
            if in_begin < offset[i]:
                pad_size = np.min([offset[i] - in_begin, offset[i + 1] - offset[i]])
                if padding_trainable:
                    sub_w = padding_data[j : j + pad_size, :]
                    col[
                        offset[i] : offset[i] + pad_size,
                        j * channels : (j + 1) * channels,
                    ] = sub_w
                out_begin = offset[i] + pad_size
                in_begin = offset[i]

            if in_end > offset[i + 1]:
                pad_size = np.min([in_end - offset[i + 1], offset[i + 1] - offset[i]])
                if padding_trainable:
                    sub_w = padding_data[
                        begin_pad + context_start + j - pad_size : begin_pad
                        + context_start
                        + j,
                        :,
                    ]
                    col[
                        offset[i + 1] - pad_size : offset[i + 1],
                        j * channels : (j + 1) * channels,
                    ] = sub_w
                in_end = offset[i + 1]
                out_end = offset[i + 1] - pad_size
            if in_end <= in_begin:
                continue
            in_sub = x[in_begin:in_end, :]
            col[out_begin:out_end, j * channels : (j + 1) * channels] += in_sub
    return np.dot(col, filter)


class TestSeqProjectMusa(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "sequence_conv"
        np.random.seed(2026)

        x = np.random.uniform(0.1, 1, self.input_size).astype("float32")
        w = np.random.uniform(
            0.1,
            1,
            [self.context_length * self.input_size[1], self.output_representation],
        ).astype("float32")

        begin_pad = np.max([0, -self.context_start])
        end_pad = np.max([0, self.context_start + self.context_length - 1])
        total_pad = begin_pad + end_pad
        padding_data = np.random.uniform(0.1, 1, [total_pad, self.input_size[1]]).astype(
            "float32"
        )
        self.inputs = {"X": (x, self.lod), "Filter": w}
        self.inputs_val = ["X", "Filter"]
        self.inputs_val_no_x = ["Filter"]
        self.inputs_val_no_f = ["X"]

        if total_pad != 0:
            self.inputs["PaddingData"] = padding_data
            self.inputs_val = ["X", "PaddingData", "Filter"]
            self.inputs_val_no_x = ["PaddingData", "Filter"]
            self.inputs_val_no_f = ["PaddingData", "X"]

        self.attrs = {
            "contextStart": self.context_start,
            "contextLength": self.context_length,
            "paddingTrainable": self.padding_trainable,
            "contextStride": self.context_stride,
        }
        self.outputs = {
            "Out": seqconv(
                x,
                self.lod,
                w,
                self.context_length,
                self.context_start,
                self.padding_trainable,
                padding_data,
            )
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad_input(self):
        self.check_grad(
            ["X"],
            "Out",
            max_relative_error=0.05,
            no_grad_set=set(self.inputs_val_no_x),
            check_dygraph=False,
        )

    def test_check_grad_filter(self):
        self.check_grad(
            ["Filter"],
            "Out",
            max_relative_error=0.05,
            no_grad_set=set(self.inputs_val_no_f),
            check_dygraph=False,
        )

    def init_test_case(self):
        self.input_size = [11, 23]
        self.context_start = 0
        self.context_length = 1
        self.padding_trainable = False
        self.context_stride = 1
        offset_lod = [[0, 4, 5, 8, self.input_size[0]]]
        self.lod = [[offset_lod[0][i + 1] - offset_lod[0][i] for i in range(len(offset_lod[0]) - 1)]]
        self.output_representation = 8


class TestSeqProjectPaddingMusa(TestSeqProjectMusa):
    def init_test_case(self):
        self.input_size = [11, 50]
        self.context_start = -1
        self.context_length = 3
        self.padding_trainable = True
        self.context_stride = 1
        offset_lod = [[0, 4, 5, 8, self.input_size[0]]]
        self.lod = [[offset_lod[0][i + 1] - offset_lod[0][i] for i in range(len(offset_lod[0]) - 1)]]
        self.output_representation = 8

    def test_check_grad_padding_data(self):
        self.check_grad(
            ["PaddingData"],
            "Out",
            max_relative_error=0.01,
            no_grad_set={"X", "Filter"},
            check_dygraph=False,
        )


class TestSparseConvExtraOpMusa(unittest.TestCase):
    def test_sparse_conv3d_official_case(self):
        paddle.set_device("musa")
        kernel = paddle.to_tensor(
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="float32"),
            stop_gradient=False,
        )
        dense_kernel = paddle.reshape(kernel, [1, 3, 3, 1, 1])
        indices = paddle.to_tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]],
            dtype="int32",
        )
        values = paddle.to_tensor([[1], [2], [3], [4]], dtype="float32")
        sparse_input = core.eager.sparse_coo_tensor(
            indices, values, [1, 1, 3, 4, 1], False
        )
        out = paddle.sparse.nn.functional.conv3d(
            sparse_input,
            dense_kernel,
            bias=paddle.to_tensor([1], dtype="float32"),
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NDHWC",
        )
        out.backward(out)
        out = paddle.sparse.coalesce(out)
        np.testing.assert_array_equal([[5], [11]], out.values().numpy())
        self.assertIsNotNone(kernel.grad)

    def test_sparse_maxpool3d_official_case(self):
        paddle.set_device("musa")
        paddle.seed(0)
        dense_x = paddle.randn((1, 4, 4, 4, 4))
        dense_x.stop_gradient = False
        sparse_x = dense_x.to_sparse_coo(4)
        sparse_out = paddle.sparse.nn.functional.max_pool3d(
            sparse_x,
            [3, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
        )
        out = sparse_out.to_dense()
        out.backward(out)

        dense_ref = paddle.assign(dense_x.detach())
        dense_ref.stop_gradient = False
        dense_out = paddle.nn.functional.max_pool3d(
            dense_ref,
            [3, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            data_format="NDHWC",
        )
        dense_out.backward(dense_out)
        np.testing.assert_allclose(dense_out.numpy(), out.numpy())
        np.testing.assert_allclose(dense_ref.grad.numpy(), dense_x.grad.numpy())

    def test_slow_conv2d_dilated_official_case(self):
        paddle.set_device("musa")
        paddle.seed(2026)
        x = paddle.randn([2, 3, 7, 7], dtype="float32")
        weight = paddle.randn([4, 3, 3, 3], dtype="float32")
        bias = paddle.randn([4], dtype="float32")
        x.stop_gradient = False
        weight.stop_gradient = False
        bias.stop_gradient = False

        out = paddle._C_ops.slow_conv2d_dilated(
            x,
            weight,
            bias,
            [1, 1],
            [1, 1],
            "EXPLICIT",
            [2, 2],
            1,
            "NCHW",
        )
        out.backward(out)

        ref_x = paddle.assign(x.detach())
        ref_weight = paddle.assign(weight.detach())
        ref_bias = paddle.assign(bias.detach())
        ref_x.stop_gradient = False
        ref_weight.stop_gradient = False
        ref_bias.stop_gradient = False
        ref = paddle.nn.functional.conv2d(
            ref_x,
            ref_weight,
            bias=ref_bias,
            stride=[1, 1],
            padding=[1, 1],
            dilation=[2, 2],
            groups=1,
            data_format="NCHW",
        )
        ref.backward(ref)

        np.testing.assert_allclose(ref.numpy(), out.numpy(), rtol=1e-3, atol=2e-2)
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)

    def test_slow_conv3d_dilated_official_case(self):
        paddle.set_device("musa")
        paddle.seed(2026)
        x = paddle.randn([2, 2, 5, 5, 5], dtype="float32")
        weight = paddle.randn([3, 2, 3, 3, 3], dtype="float32")
        bias = paddle.randn([3], dtype="float32")
        x.stop_gradient = False
        weight.stop_gradient = False
        bias.stop_gradient = False

        out = paddle._C_ops.slow_conv3d_dilated(
            x,
            weight,
            bias,
            [1, 1, 1],
            [1, 1, 1],
            "EXPLICIT",
            1,
            [2, 2, 2],
            "NCDHW",
        )
        out.backward(out)

        ref_x = paddle.assign(x.detach())
        ref_weight = paddle.assign(weight.detach())
        ref_bias = paddle.assign(bias.detach())
        ref_x.stop_gradient = False
        ref_weight.stop_gradient = False
        ref_bias.stop_gradient = False
        ref = paddle.nn.functional.conv3d(
            ref_x,
            ref_weight,
            bias=ref_bias,
            stride=[1, 1, 1],
            padding=[1, 1, 1],
            dilation=[2, 2, 2],
            groups=1,
            data_format="NCDHW",
        )
        ref.backward(ref)

        np.testing.assert_allclose(ref.numpy(), out.numpy(), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(ref_x.grad.numpy(), x.grad.numpy(), rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(
            ref_weight.grad.numpy(), weight.grad.numpy(), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            ref_bias.grad.numpy(), bias.grad.numpy(), rtol=1e-3, atol=1e-5
        )

    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "conv3d_coo",
            "conv3d_coo_grad",
            "maxpool_coo_grad",
            "sequence_conv",
            "sequence_conv_grad",
            "slow_conv2d_dilated",
            "slow_conv2d_dilated_grad",
            "slow_conv3d_dilated",
            "slow_conv3d_dilated_grad",
            "resnet_unit_grad",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
