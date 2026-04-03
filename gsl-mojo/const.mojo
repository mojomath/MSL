# ===----------------------------------------------------------------------=== #
# gsl-bindings: GSL ported to Mojo
# Distributed under the GNU General Public License v3.
# See LICENSE for more information.
# ===----------------------------------------------------------------------=== #
"""
Mathematical constants.

These are standard constants used throughout GSL.
"""


# ===----------------------------------------------------------------------=== #
# Mathematical constants (from gsl_math.h, gsl_const_num.h)
# ===----------------------------------------------------------------------=== #

comptime M_E: f64 = 2.71828182845904523536028747135
"""Base of natural logarithm (e)."""

comptime M_LOG2E: f64 = 1.44269504088896340735992468100
"""Log base 2 of e."""

comptime M_LOG10E: f64 = 0.43429448190325182765112891892
"""Log base 10 of e."""

comptime M_SQRT2: f64 = 1.41421356237309504880168872421
"""Square root of 2."""

comptime M_SQRT1_2: f64 = 0.70710678118654752440084436210
"""Square root of 1/2."""

comptime M_SQRT3: f64 = 1.73205080756887729352744634151
"""Square root of 3."""

comptime M_PI: f64 = 3.14159265358979323846264338328
"""Pi."""

comptime M_PI_2: f64 = 1.57079632679489661923132169164
"""Pi/2."""

comptime M_PI_4: f64 = 0.78539816339744830961566084582
"""Pi/4."""

comptime M_SQRTPI: f64 = 1.77245385090551602729816748334
"""Square root of pi."""

comptime M_2_SQRTPI: f64 = 1.12837916709551257389615890312
"""2/sqrt(pi)."""

comptime M_1_PI: f64 = 0.31830988618379067153776752675
"""1/pi."""

comptime M_2_PI: f64 = 0.63661977236758134307553505349
"""2/pi."""

comptime M_LN10: f64 = 2.30258509299404568401799145468
"""Natural log of 10."""

comptime M_LN2: f64 = 0.69314718055994530941723212146
"""Natural log of 2."""

comptime M_LNPI: f64 = 1.14472988584940017414342735135
"""Natural log of pi."""

comptime M_EULER: f64 = 0.57721566490153286060651209008
"""Euler-Mascheroni constant."""


# ===----------------------------------------------------------------------=== #
# Machine constants (from gsl_machine.h)
# ===----------------------------------------------------------------------=== #

comptime GSL_DBL_EPSILON: f64 = 2.2204460492503131e-16
"""Double precision machine epsilon."""

comptime GSL_SQRT_DBL_EPSILON: f64 = 1.4901161193847656e-08
"""Square root of double precision machine epsilon."""

comptime GSL_DBL_MIN: f64 = 2.2250738585072014e-308
"""Smallest normalized double."""

comptime GSL_DBL_MAX: f64 = 1.7976931348623157e+308
"""Largest representable double."""

comptime GSL_FLT_EPSILON: f64 = 1.1920928955078125e-07
"""Single precision machine epsilon."""

comptime GSL_SQRT_FLT_EPSILON: c_float = 3.4526698300124393e-04
"""Square root of single precision machine epsilon."""

comptime GSL_FLT_MIN: c_float = 1.1754943508222875e-38
"""Smallest normalized float."""

comptime GSL_FLT_MAX: c_float = 3.4028234663852886e+38
"""Largest representable float."""


# ===----------------------------------------------------------------------=== #
# IEEE floating point (from gsl_ieee_utils.h)
# ===----------------------------------------------------------------------=== #

comptime GSL_IEEE_SINGLE: Int = 1
comptime GSL_IEEE_DOUBLE: Int = 2
comptime GSL_IEEE_EXTENDED: Int = 3


# ===----------------------------------------------------------------------=== #
# Precision modes (from gsl_mode.h)
# ===----------------------------------------------------------------------=== #

comptime GSL_PRECISION_DOUBLE: Int = 0
comptime GSL_PRECISION_SINGLE: Int = 1
comptime GSL_PRECISION_APPROX: Int = 2


# ===----------------------------------------------------------------------=== #
# Range checking (from gsl_check_range.h)
# ===----------------------------------------------------------------------=== #

comptime GSL_RANGE_CHECK: Int = 1
