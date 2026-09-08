# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


class DML:
    """
    DML specific subgraph expansions
    """

    def make_gated_add(self, name, root_input, scaled_input, gate, shape):
        #      scaled_input   gate
        #             \       /
        #              \     /
        #               Mul      root_input
        #                 \       /
        #                  \     /
        #                   Add
        mul_name = f"{name}/Mul"
        self.make_mul(mul_name, [scaled_input, gate], self.io_dtype, shape=shape)
        self.make_add(name, [root_input, f"{mul_name}/output_0"], self.io_dtype, shape=shape)
