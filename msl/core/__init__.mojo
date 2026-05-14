# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
Core types, error codes, and math utilities.
"""

from .block import Block, block_alloc, block_calloc, block_size, block_data
from .errno import MSL_SUCCESS, MSL_FAILURE, MSL_ENOMEM, MSL_EINVAL
from .minmax import min, max
from .nan import msl_isnan, msl_isinf
from .pow_int import msl_pow_int
