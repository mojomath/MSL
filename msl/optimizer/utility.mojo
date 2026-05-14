# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Shivasankar K.A.
"""Result types for scalar root-finding and minimization."""

from msl.core.errno import MSL_SUCCESS


struct RootResult(Copyable, Movable):
    """Result of a scalar root-finding operation."""

    var root: Float64
    """Root estimate."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations."""
    var success: Bool
    """True if converged within tolerance."""
    var errno: Int
    """Error code (0 = success)."""

    def __init__(
        out self,
        root: Float64 = 0.0,
        nit: Int = 0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.root = root
        self.nit = nit
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.root = copy.root
        self.nit = copy.nit
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit take: Self):
        self.root = take.root
        self.nit = take.nit
        self.nfev = take.nfev
        self.success = take.success
        self.errno = take.errno


struct MinResult(Copyable, Movable):
    """Result of a scalar minimization operation."""

    var x: Float64
    """Location of the minimum."""
    var fun: Float64
    """Function value at the minimum."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations."""
    var success: Bool
    """True if converged within tolerance."""
    var errno: Int
    """Error code (0 = success)."""

    def __init__(
        out self,
        x: Float64 = 0.0,
        fun: Float64 = 0.0,
        nit: Int = 0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.x = x
        self.fun = fun
        self.nit = nit
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.fun = copy.fun
        self.nit = copy.nit
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit take: Self):
        self.x = take.x
        self.fun = take.fun
        self.nit = take.nit
        self.nfev = take.nfev
        self.success = take.success
        self.errno = take.errno
