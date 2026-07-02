import sys
import unittest
from pathlib import Path

import numpy as np
import paddle

sys.path.append(str(Path(__file__).resolve().parents[4] / "Paddle" / "test" / "legacy_test"))
from op import Operator
from paddle.base import core


def adam_step_sparse(inputs, attributes, height, rows, row_numel, np_grad, lazy_mode):
    param = inputs["Param"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    moment2_max = inputs["Moment2Max"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    beta1 = attributes["beta1"]
    beta2 = attributes["beta2"]
    epsilon = attributes["epsilon"]
    amsgrad = attributes["amsgrad"]

    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    moment2_max_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])

    def update_row(row_id, update_value):
        moment1_out[row_id] = beta1 * moment1[row_id] + (1 - beta1) * update_value
        moment2_out[row_id] = beta2 * moment2[row_id] + (1 - beta2) * np.square(
            update_value
        )
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)

        if amsgrad:
            moment2_max_out[row_id] = np.maximum(moment2_out[row_id], moment2_max[row_id])
            param_out[row_id] = param[row_id] - lr_t * (
                moment1_out[row_id] / (np.sqrt(moment2_max_out[row_id]) + epsilon)
            )
        else:
            moment2_max_out[row_id] = np.empty_like(moment2_out[row_id])
            param_out[row_id] = param[row_id] - lr_t * (
                moment1_out[row_id] / (np.sqrt(moment2_out[row_id]) + epsilon)
            )

    if lazy_mode:
        for idx, row_id in enumerate(rows):
            update_row(row_id, np_grad[idx])
    else:
        for row_id in range(param_out.shape[0]):
            update_value = np.zeros(np_grad[0].shape).astype("float32")
            if row_id in rows:
                update_value = np_grad[rows.index(row_id)]
            update_row(row_id, update_value)

    return param_out, moment1_out, moment2_out, moment2_max_out


class TestSparseAdamOfficialMusa(unittest.TestCase):
    def check_with_place(self, place, lazy_mode):
        scope = core.Scope()
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = np.array([beta1**10]).astype("float32")
        beta2_pow = np.array([beta2**10]).astype("float32")
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2Max": np.zeros((height, row_numel)).astype("float32"),
            "Beta1Pow": beta1_pow,
            "Beta2Pow": beta2_pow,
            "LearningRate": np.full((1), 2.0).astype("float64"),
        }
        attrs = {
            "epsilon": epsilon,
            "beta1": beta1,
            "beta2": beta2,
            "min_row_size_to_use_multithread": 2,
            "amsgrad": False,
        }

        grad_selected_rows = scope.var("Grad").get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        grad_np = np.ones((len(rows), row_numel)).astype("float32")
        grad_np[0, 0] = 2.0
        grad_np[2, 8] = 4.0
        grad_selected_rows.get_tensor().set(grad_np, place)

        param_out, mom1, mom2, mom2_max = adam_step_sparse(
            dense_inputs, attrs, height, rows, row_numel, grad_np, lazy_mode
        )
        outputs = {
            "ParamOut": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2,
            "Moment2MaxOut": mom2_max,
            "Beta1PowOut": beta1_pow * beta1,
            "Beta2PowOut": beta2_pow * beta2,
        }

        op_args = {"Grad": "Grad", "lazy_mode": lazy_mode}
        for key, np_array in dense_inputs.items():
            scope.var(key).get_tensor().set(np_array, place)
            op_args[key] = key
        for key in outputs:
            scope.var(key).get_tensor().set(np.zeros((height, row_numel)).astype("float32") if key.endswith("Out") and key not in ["Beta1PowOut", "Beta2PowOut"] else np.zeros((1)).astype("float32"), place)
            op_args[key] = key
        op_args.update(attrs)

        adam_op = Operator("adam", **op_args)
        adam_op.run(scope, place)

        for key, expected in outputs.items():
            if key == "Moment2MaxOut":
                continue
            np.testing.assert_allclose(np.array(scope.var(key).get_tensor()), expected, atol=2e-5)

    def test_sparse_adam(self):
        place = paddle.CustomPlace("musa", 0)
        for lazy_mode in (True, False):
            self.check_with_place(place, lazy_mode)


class TestSelectedRowsGradExtraOpMusa(unittest.TestCase):
    def test_registered(self):
        paddle.set_device("musa")
        kernels = core._get_registered_phi_kernels()
        for op_name in [
            "adam_dense_param_sparse_grad",
            "clip_by_norm_sr",
            "dgc_clip_by_norm_sr",
            "lookup_table_grad_sr",
            "lookup_table_sparse_grad_sr",
        ]:
            self.assertIn(op_name, kernels)
            self.assertTrue(any("musa" in kernel for kernel in kernels[op_name]))


if __name__ == "__main__":
    unittest.main()
