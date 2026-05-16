# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
BLAS wrapper for MSL Vector and Matrix types.

Provides type-safe BLAS operations on MSL's `Vector` and `Matrix` structs
by unwrapping their raw pointers and delegating to mojoBLAS.

Level 1 (vector-vector):
  blas_dot    - dot product: x · y
  blas_nrm2   - Euclidean norm: ||x||_2
  blas_asum   - sum of absolute values: sum|x_i|
  blas_axpy   - AXPY: y += alpha * x
  blas_scal   - scale: x *= alpha
  blas_copy   - copy: y = x
  blas_swap   - swap: x ↔ y

Level 2 (matrix-vector):
  blas_gemv   - general matrix-vector: y = alpha*A*x + beta*y

Level 3 (matrix-matrix):
  blas_gemm   - general matrix-matrix: C = alpha*A*B + beta*C
"""

from mojoBLAS.level1 import dot, nrm2, asum, axpy, scal, copy, vswap
from mojoBLAS.level2 import gemv
from mojoBLAS.level3 import gemm

from msl.vector import Vector
from msl.matrix import Matrix
from msl.core.errno import MSL_SUCCESS, MSL_EINVAL


# ===----------------------------------------------------------------------=== #
# Level 1 - vector-vector
# ===----------------------------------------------------------------------=== #


def blas_dot(x: Vector, y: Vector) -> Float64:
    """Dot product x · y.

    Args:
        x: First vector.
        y: Second vector (must have same size as x).

    Returns:
        Scalar dot product, or 0.0 if sizes differ.
    """
    if x.size != y.size:
        return 0.0
    return dot[DType.float64](
        x.size,
        x.ptr_read(),
        x.stride,
        y.ptr_read(),
        y.stride,
    )


def blas_nrm2(x: Vector) -> Float64:
    """Euclidean norm ||x||_2.

    Args:
        x: Input vector.

    Returns:
        Euclidean norm.
    """
    return nrm2[DType.float64](
        x.size,
        x.ptr_read(),
        x.stride,
    )


def blas_asum(x: Vector) -> Float64:
    """Sum of absolute values sum|x_i|.

    Args:
        x: Input vector.

    Returns:
        Sum of absolute values.
    """
    return asum[DType.float64](
        x.size,
        x.ptr_read(),
        x.stride,
    )


def blas_axpy(alpha: Float64, x: Vector, mut y: Vector):
    """AXPY: y += alpha * x.

    Args:
        alpha: Scalar multiplier.
        x: Source vector.
        y: Destination vector (updated in-place, same size as x).
    """
    if x.size != y.size:
        return
    axpy[DType.float64](
        x.size,
        alpha,
        x.ptr_read(),
        x.stride,
        y.ptr_mut(),
        y.stride,
    )


def blas_scal(alpha: Float64, mut x: Vector):
    """Scale: x *= alpha.

    Args:
        alpha: Scalar multiplier.
        x: Vector to scale (updated in-place).
    """
    scal[DType.float64](
        x.size,
        alpha,
        x.ptr_mut(),
        x.stride,
    )


def blas_copy(x: Vector, mut y: Vector):
    """Copy: y = x.

    Args:
        x: Source vector.
        y: Destination vector (must have same size as x).
    """
    if x.size != y.size:
        return
    copy[DType.float64](
        x.size,
        x.ptr_read(),
        x.stride,
        y.ptr_mut(),
        y.stride,
    )


def blas_swap(mut x: Vector, mut y: Vector):
    """Swap: x ↔ y.

    Args:
        x: First vector (modified in-place).
        y: Second vector (modified in-place, same size as x).
    """
    if x.size != y.size:
        return
    vswap[DType.float64](
        x.size,
        x.ptr_mut(),
        x.stride,
        y.ptr_mut(),
        y.stride,
    )


# ===----------------------------------------------------------------------=== #
# Level 2 - matrix-vector
# ===----------------------------------------------------------------------=== #


def blas_gemv(
    a: Matrix,
    x: Vector,
    mut y: Vector,
    alpha: Float64 = 1.0,
    beta: Float64 = 0.0,
    trans: Bool = False,
):
    """General matrix-vector multiply: y = alpha * op(A) * x + beta * y.

    Args:
        a: Matrix A of shape (m, n).
        x: Input vector (length n if not transposed, m if transposed).
        y: Output vector (length m if not transposed, n if transposed).
        alpha: Scalar multiplier for A*x.
        beta: Scalar multiplier for y.
        trans: If True, use A^T instead of A.
    """
    var t = "T" if trans else "N"
    var m = a.s1
    var n = a.s2
    gemv[DType.float64](
        t, m, n,
        alpha,
        a.ptr_read(),
        a.tda,
        x.ptr_read(),
        x.stride,
        beta,
        y.ptr_mut(),
        y.stride,
    )


# ===----------------------------------------------------------------------=== #
# Level 3 - matrix-matrix
# ===----------------------------------------------------------------------=== #


def blas_gemm(
    a: Matrix,
    b: Matrix,
    mut c: Matrix,
    alpha: Float64 = 1.0,
    beta: Float64 = 0.0,
    trans_a: Bool = False,
    trans_b: Bool = False,
):
    """General matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C.

    Args:
        a: Left matrix.
        b: Right matrix.
        c: Output matrix (updated in-place).
        alpha: Scalar multiplier for A*B.
        beta: Scalar multiplier for C.
        trans_a: If True, use A^T.
        trans_b: If True, use B^T.
    """
    var ta = "T" if trans_a else "N"
    var tb = "T" if trans_b else "N"
    var m = a.s1 if not trans_a else a.s2
    var k = a.s2 if not trans_a else a.s1
    var n = b.s2 if not trans_b else b.s1
    gemm[DType.float64](
        ta, tb,
        m, n, k,
        alpha,
        a.ptr_read(),
        a.tda,
        b.ptr_read(),
        b.tda,
        beta,
        c.ptr_mut(),
        c.tda,
    )
