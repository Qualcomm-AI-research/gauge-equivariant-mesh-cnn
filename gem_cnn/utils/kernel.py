# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
from itertools import product

import numpy as np


def create_kernel_basis(l_in, l_out, band_limit=None):
    """
    Create basis tensor, for one radial basis function.

    d_in/out is dimenion of irrep order l_in/out.
    num_basis is number of basis elements of intertwiner space.

    theta_freq_dims is a list of theta frequency dimensions that needs to be selected from
    Fourier transformed input. This is to avoid sparsity in kernel.

    Frequency is the theta frequency of each basis-intertwiner. Used for band-limiting.

    :param l_in: int
    :param l_out: int
    :param band_limit: int or None
    :return bases: [num_basis, d_out, num_theta_freq_dims, d_in]
    :return theta_freq_dims: [num_theta_freq_dims]
    :return frequencies: [num_basis]
    """
    if l_in == 0 and l_out == 0:
        bases = np.ones((1, 1, 1, 1))
        frequencies = [0]
        theta_freq_dims = [0]
    elif l_in == 0 and l_out != 0:
        bases = np.array([[1, 0, 0, 1], [0, -1, 1, 0]]).reshape((2, 2, 2, 1))
        frequencies = [l_out, l_out]
        theta_freq_dims = [2 * l_out - 1, 2 * l_out]
    elif l_in != 0 and l_out == 0:
        bases = np.array([[1, 0, 0, 1], [0, -1, 1, 0]]).reshape((2, 1, 2, 2))
        frequencies = [l_in, l_in]
        theta_freq_dims = [2 * l_in - 1, 2 * l_in]
    else:
        basis_plus = np.array([[0, 1, -1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, -1, 1, 0]]).reshape(
            (2, 2, 2, 2)
        )
        basis_min_pos = np.array([[0, 1, 1, 0, -1, 0, 0, 1], [1, 0, 0, -1, 0, 1, 1, 0]]).reshape(
            (2, 2, 2, 2)
        )
        basis_min_neg = np.array([[0, -1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, -1, 0]]).reshape(
            (2, 2, 2, 2)
        )
        basis_zero = np.array([[1, 0, 0, 1], [0, -1, 1, 0]]).reshape((2, 2, 1, 2))

        l_plus = l_out + l_in
        l_min = np.abs(l_out - l_in)
        if l_out == l_in:
            bases = np.concatenate(
                [
                    np.concatenate(
                        [basis_zero, np.zeros((2, 2, 2, 2), dtype=np.int32)], 2
                    ),  # [2, 2, 3, 2]
                    np.concatenate(
                        [np.zeros((2, 2, 1, 2), dtype=np.int32), basis_plus], 2
                    ),  # [2, 2, 3, 2]
                ],
                0,
            )  # [4, 2, 3, 2]
            frequencies = [0, 0, l_plus, l_plus]
            theta_freq_dims = [0, 2 * l_plus - 1, 2 * l_plus]
        else:
            basis_min = basis_min_pos if l_out > l_in else basis_min_neg

            bases = np.concatenate(
                [
                    np.concatenate([basis_min, np.zeros((2, 2, 2, 2), dtype=np.int32)], 2),
                    np.concatenate([np.zeros((2, 2, 2, 2), dtype=np.int32), basis_plus], 2),
                ],
                0,
            )
            frequencies = [l_min, l_min, l_plus, l_plus]
            theta_freq_dims = [2 * l_min - 1, 2 * l_min, 2 * l_plus - 1, 2 * l_plus]

    theta_freq_dims = np.array(theta_freq_dims)
    frequencies = np.array(frequencies)

    # Apply band limit
    if band_limit is not None:
        bases_to_filter = np.array([f > band_limit for f in frequencies])
        bases = bases[~bases_to_filter]
        frequencies = [f for f in frequencies if f <= band_limit]

        # If some theta freqs are not used, remove from basis (axis 2) and selector list
        theta_freq_dims_to_filter = (bases == 0).all(3).all(1).all(0)
        bases = bases[:, :, ~theta_freq_dims_to_filter]
        theta_freq_dims = theta_freq_dims[~theta_freq_dims_to_filter]

    return bases, theta_freq_dims, frequencies


def build_kernel(in_order, out_order, band_limit=None):
    """

    n_bases = 4 * (order_in * order_out) + 2 * (order_in + order_out) + 1
    d_in = 2 * order_in + 1
    d_out = 2 * order_out + 1

    :param in_order: int
    :param out_order: int
    :param band_limit: optional int (default order_in + order_out)
    :return: [n_bases, 2 * band_limit + 1, d_out, d_in]
    """
    band_limit = band_limit or in_order + out_order
    n_bases = 4 * (in_order * out_order) + 2 * (in_order + out_order) + 1
    kernel = np.zeros((n_bases, band_limit * 2 + 1, out_order * 2 + 1, in_order * 2 + 1))
    idx = 0
    for l_in, l_out in product(range(in_order + 1), range(out_order + 1)):
        bases, theta_freq_dims, frequencies = create_kernel_basis(
            l_in, l_out, band_limit=band_limit
        )
        out_slice = slice(1) if l_out == 0 else slice(2 * l_out - 1, 2 * l_out + 1)
        in_slice = slice(1) if l_in == 0 else slice(2 * l_in - 1, 2 * l_in + 1)
        kernel[idx : idx + len(bases), theta_freq_dims, out_slice, in_slice] = bases.transpose(
            [0, 2, 1, 3]
        )
        idx += len(bases)
    return kernel
