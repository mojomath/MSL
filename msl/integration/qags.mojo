# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/qags.c, integration/qelg.c
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough
#
# Modifications:
# Copyright (C) 2026 Shivasankar K.A.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------------=== #
"""
QAGS — Adaptive integration with Wynn epsilon extrapolation.

Like QAG but applies the Wynn epsilon algorithm to accelerate convergence
for smooth but slowly-converging integrands, and handles singularities
better via extrapolation.
"""

from std.math import abs
from std.collections import InlineArray

from msl.core.const import MSL_DBL_EPSILON, MSL_DBL_MIN, MSL_DBL_MAX
from .workspace import IntegrationWorkspace, IntegrationResult, QKResult
from .qk21 import qk21


def _subinterval_too_small(a1: Float64, a2: Float64, b2: Float64) -> Bool:
    var tmp = (1.0 + 100.0 * MSL_DBL_EPSILON) * (abs(a2) + 1000.0 * MSL_DBL_MIN)
    return abs(a1) <= tmp and abs(b2) <= tmp


# ===----------------------------------------------------------------------=== #
# Wynn epsilon extrapolation table
# ===----------------------------------------------------------------------=== #


struct ExtrapolationTable(Movable):
    var rlist2: InlineArray[Float64, 52]
    var res3la: InlineArray[Float64, 3]
    var n: Int
    var nres: Int

    def __init__(out self):
        self.rlist2 = InlineArray[Float64, 52](fill=0.0)
        self.res3la = InlineArray[Float64, 3](fill=0.0)
        self.n = 0
        self.nres = 0

    def __init__(out self, *, deinit take: Self):
        self.rlist2 = take.rlist2
        self.res3la = take.res3la
        self.n = take.n
        self.nres = take.nres

    def append(mut self, val: Float64):
        self.rlist2[self.n] = val
        self.n += 1

    def qelg(mut self) -> Tuple[Float64, Float64]:
        """Wynn epsilon algorithm. Returns (result, abserr)."""
        var n = self.n - 1
        var current = self.rlist2[n]

        if n < 2:
            var abserr = max(MSL_DBL_MAX, 5.0 * MSL_DBL_EPSILON * abs(current))
            self.nres += 1
            return (current, abserr)

        var limexp = 49
        self.rlist2[n + 2] = self.rlist2[n]
        self.rlist2[n] = MSL_DBL_MAX

        var newelm = n // 2
        var n_final = n

        var best_abserr = MSL_DBL_MAX
        var best_result = current

        for i in range(newelm):
            var res = self.rlist2[n - 2 * i + 2]
            var e0 = self.rlist2[n - 2 * i - 2]
            var e1 = self.rlist2[n - 2 * i - 1]
            var e2 = res

            var e1abs = abs(e1)
            var delta2 = e2 - e1
            var err2 = abs(delta2)
            var tol2 = max(abs(e2), e1abs) * MSL_DBL_EPSILON
            var delta3 = e1 - e0
            var err3 = abs(delta3)
            var tol3 = max(e1abs, abs(e0)) * MSL_DBL_EPSILON

            if err2 <= tol2 and err3 <= tol3:
                best_result = res
                self.nres += 1
                if self.nres < 3:
                    self.res3la[self.nres] = best_result
                    best_abserr = MSL_DBL_MAX
                else:
                    best_abserr = (
                        abs(best_result - self.res3la[2])
                        + abs(best_result - self.res3la[1])
                        + abs(best_result - self.res3la[0])
                    )
                    self.res3la[0] = self.res3la[1]
                    self.res3la[1] = self.res3la[2]
                    self.res3la[2] = best_result
                best_abserr = max(
                    best_abserr, 5.0 * MSL_DBL_EPSILON * abs(best_result)
                )
                return (best_result, best_abserr)

            var e3 = self.rlist2[n - 2 * i]
            self.rlist2[n - 2 * i] = e1
            var delta1 = e1 - e3
            var err1 = abs(delta1)
            var tol1 = max(e1abs, abs(e3)) * MSL_DBL_EPSILON

            if err1 <= tol1 or err2 <= tol2 or err3 <= tol3:
                n_final = 2 * i
                break

            var ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3

            if abs(ss * e1) <= 0.0001:
                n_final = 2 * i
                break

            res = e1 + 1.0 / ss
            self.rlist2[n - 2 * i] = res

            var error = err2 + abs(res - e2) + err3
            if error <= best_abserr:
                best_abserr = error
                best_result = res

        # Table compaction
        var n_orig = n
        if n_final == limexp:
            n_final = 2 * (limexp // 2)

        if n_orig % 2 == 1:
            for i in range(newelm + 1):
                self.rlist2[1 + 2 * i] = self.rlist2[2 * i + 3]
        else:
            for i in range(newelm + 1):
                self.rlist2[2 * i] = self.rlist2[2 * i + 2]

        if n_orig != n_final:
            for i in range(n_final + 1):
                self.rlist2[i] = self.rlist2[n_orig - n_final + i]

        self.n = n_final + 1

        self.nres += 1
        var abserr: Float64
        if self.nres < 3:
            self.res3la[self.nres] = best_result
            abserr = MSL_DBL_MAX
        else:
            abserr = (
                abs(best_result - self.res3la[2])
                + abs(best_result - self.res3la[1])
                + abs(best_result - self.res3la[0])
            )
            self.res3la[0] = self.res3la[1]
            self.res3la[1] = self.res3la[2]
            self.res3la[2] = best_result

        abserr = max(abserr, 5.0 * MSL_DBL_EPSILON * abs(best_result))
        return (best_result, abserr)


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def qags[
    integrand: def(Float64) capturing -> Float64
](
    a: Float64,
    b: Float64,
    epsabs: Float64,
    epsrel: Float64,
    limit: Int = 50,
) -> IntegrationResult:
    """Adaptive integration with Wynn epsilon extrapolation over [a, b].

    Uses QK21 as the base rule. Applies the Wynn epsilon algorithm to
    accelerate convergence, making it effective for smooth integrands
    and integrands with endpoint singularities.

    Parameters:
        integrand: Function to integrate, must have `capturing` effect.

    Args:
        a: Lower limit.
        b: Upper limit.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.
        limit: Maximum number of subintervals (default 50).

    Returns:
        IntegrationResult with val (integral estimate) and err (error bound).
    """
    var result = IntegrationResult()

    var r0 = qk21[integrand](a, b)
    var area = r0.result
    var errsum = r0.abserr
    var resabs0 = r0.resabs
    var positive_integrand = (
        abs(area) >= (1.0 - 50.0 * MSL_DBL_EPSILON) * resabs0
    )

    var tolerance = max(epsabs, epsrel * abs(area))
    var round_off = 100.0 * MSL_DBL_EPSILON * resabs0

    if errsum <= round_off and errsum > tolerance:
        result.val = area
        result.err = errsum
        return result^

    if (errsum <= tolerance and errsum != r0.resasc) or errsum == 0.0:
        result.val = area
        result.err = errsum
        return result^

    var ws = IntegrationWorkspace(limit)
    ws.initialise(a, b, area, errsum)

    var table = ExtrapolationTable()
    table.append(area)

    var res_ext = area
    var err_ext = MSL_DBL_MAX
    var error_over_large_intervals = errsum
    var ertest = tolerance
    var correc: Float64 = 0.0
    var ktmin: Int = 0
    var roundoff_type1: Int = 0
    var roundoff_type2: Int = 0
    var roundoff_type3: Int = 0
    var error_type: Int = 0
    var error_type2 = False
    var extrapolate = False
    var disallow_extrapolation = False

    for iteration in range(1, limit):
        var a_i: Float64
        var b_i: Float64
        var r_i: Float64
        var e_i: Float64
        a_i, b_i, r_i, e_i = ws.retrieve()
        var current_level = ws.level[ws.i] + 1

        var mid = 0.5 * (a_i + b_i)
        var r1 = qk21[integrand](a_i, mid)
        var r2 = qk21[integrand](mid, b_i)

        var area12 = r1.result + r2.result
        var error12 = r1.abserr + r2.abserr
        var last_e_i = e_i

        errsum += error12 - e_i
        area += area12 - r_i

        if r1.resasc != r1.abserr and r2.resasc != r2.abserr:
            var delta = r_i - area12
            if abs(delta) <= 1.0e-5 * abs(area12) and error12 >= 0.99 * e_i:
                if not extrapolate:
                    roundoff_type1 += 1
                else:
                    roundoff_type2 += 1
            if iteration > 10 and error12 > e_i:
                roundoff_type3 += 1

        tolerance = max(epsabs, epsrel * abs(area))

        if roundoff_type1 + roundoff_type2 >= 10 or roundoff_type3 >= 20:
            error_type = 2
        if roundoff_type2 >= 5:
            error_type2 = True
        if _subinterval_too_small(a_i, mid, b_i):
            error_type = 4

        ws.update(
            a_i, mid, r1.result, r1.abserr, mid, b_i, r2.result, r2.abserr
        )

        if errsum <= tolerance:
            break
        if error_type != 0:
            break
        if iteration >= limit - 1:
            error_type = 1
            break

        if iteration == 2:
            error_over_large_intervals = errsum
            ertest = tolerance
            table.append(area)
            continue

        if disallow_extrapolation:
            continue

        error_over_large_intervals -= last_e_i
        if current_level < ws.maximum_level:
            error_over_large_intervals += error12

        if not extrapolate:
            if ws.large_interval():
                continue
            extrapolate = True
            ws.nrmax = 1

        if not error_type2 and error_over_large_intervals > ertest:
            if ws.increase_nrmax():
                continue

        table.append(area)
        var reseps: Float64
        var abseps: Float64
        reseps, abseps = table.qelg()

        ktmin += 1
        if ktmin > 5 and err_ext < 0.001 * errsum:
            error_type = 5

        if abseps < err_ext:
            ktmin = 0
            err_ext = abseps
            res_ext = reseps
            correc = error_over_large_intervals
            ertest = max(epsabs, epsrel * abs(reseps))
            if err_ext <= ertest:
                break

        if table.n == 1:
            disallow_extrapolation = True

        if error_type == 5:
            break

        ws.reset_nrmax()
        extrapolate = False
        error_over_large_intervals = errsum

    # Choose result: extrapolated or direct sum
    var sum_result = ws.sum_results()

    if err_ext == MSL_DBL_MAX:
        result.val = sum_result
        result.err = errsum
        return result^

    if error_type2:
        err_ext += correc

    # Divergence test: if integral is not positive-dominated and tiny vs resabs0
    var max_area = max(abs(res_ext), abs(sum_result))
    if not positive_integrand and max_area < 0.01 * resabs0:
        result.val = sum_result
        result.err = errsum
        return result^

    # Fall back to direct sum if extrapolation didn't help
    if res_ext != 0.0 and sum_result != 0.0:
        if err_ext / abs(res_ext) > errsum / abs(sum_result):
            result.val = sum_result
            result.err = errsum
            return result^
    elif err_ext > errsum:
        result.val = sum_result
        result.err = errsum
        return result^
    elif sum_result == 0.0:
        result.val = res_ext
        result.err = err_ext
        return result^

    # Ratio divergence check
    if res_ext != 0.0 and sum_result != 0.0:
        var ratio = res_ext / sum_result
        if ratio < 0.01 or ratio > 100.0 or errsum > abs(sum_result):
            error_type = 6

    result.val = res_ext
    result.err = err_ext
    if error_type > 2:
        result.err = max(err_ext, errsum)
    return result^
