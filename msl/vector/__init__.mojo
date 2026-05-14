# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
1D vector with stride and linear algebra operations.
"""

from .vector import (
    Vector,
    vector_alloc,
    vector_calloc,
    vector_size,
    vector_stride,
    vector_set_zero,
    vector_set_all,
    vector_add,
    vector_sub,
    vector_scale,
    vector_axpy,
    vector_dot,
    vector_norm,
)
