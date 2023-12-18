# -*- coding: utf-8 -*-

"""
py_lets_be_rational.lets_be_rational
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure python implementation of Peter Jaeckel's LetsBeRational.

:copyright: © 2017 Gammon Capital LLC
:license: MIT, see LICENSE for more details.

About LetsBeRational:
~~~~~~~~~~~~~~~~~~~~~

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

======================================================================================
Copyright © 2013-2014 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

WARRANTY DISCLAIMER
The Software is provided "as is" without warranty of any kind, either express or implied,
including without limitation any implied warranties of condition, uninterrupted use,
merchantability, fitness for a particular purpose, or non-infringement.
======================================================================================
"""

from math import exp, fabs, log, sqrt

import numpy as np
from scipy.special import erfcx as erfcx_cody
from scipy.special import ndtr as norm_cdf
from scipy.special import ndtri as inverse_norm_cdf

from pylbr.constants import (
    DBL_EPSILON,
    DBL_MAX,
    DBL_MIN,
    DENORMALIZATION_CUTOFF,
    FOURTH_ROOT_DBL_EPSILON,
    ONE_OVER_SQRT_TWO,
    ONE_OVER_SQRT_TWO_PI,
    PI_OVER_SIX,
    SIXTEENTH_ROOT_DBL_EPSILON,
    SQRT_DBL_MAX,
    SQRT_DBL_MIN,
    SQRT_ONE_OVER_THREE,
    SQRT_PI_OVER_TWO,
    SQRT_THREE,
    SQRT_TWO_PI,
    TWO_PI,
    TWO_PI_OVER_SQRT_TWENTY_SEVEN,
)
from pylbr.exceptions import AboveMaximumException, BelowIntrinsicException
from pylbr.rationalcubic import (
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side,
    convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side,
    rational_cubic_interpolation,
)

implied_volatility_maximum_iterations = 2
asymptotic_expansion_accuracy_threshold = -10
small_t_expansion_of_normalized_black_threshold = 2 * SIXTEENTH_ROOT_DBL_EPSILON


_norm_pdf_C = np.sqrt(2 * np.pi)


def norm_pdf(x: float) -> float:
    return np.exp(-(x**2) / 2.0) / _norm_pdf_C


def _householder_factor(newton: float, halley: float, hh3: float) -> float:
    return (1 + 0.5 * halley * newton) / (1 + newton * (halley + hh3 * newton / 6))


def _compute_f_lower_map_and_first_two_derivatives(
    x: float, s: float
) -> tuple[float, float, float]:
    ax = fabs(x)
    z = SQRT_ONE_OVER_THREE * ax / s
    y = z * z
    s2 = s * s
    Phi = norm_cdf(-z)
    phi = norm_pdf(z)
    fpp = (
        PI_OVER_SIX
        * y
        / (s2 * s)
        * Phi
        * (8 * SQRT_THREE * s * ax + (3 * s2 * (s2 - 8) - 8 * x * x) * Phi / phi)
        * exp(2 * y + 0.25 * s2)
    )
    if _is_below_horizon(s):
        fp = 1
        f = 0
    else:
        Phi2 = Phi * Phi
        fp = TWO_PI * y * Phi2 * exp(y + 0.125 * s * s)
        if _is_below_horizon(x):
            f = 0
        else:
            f = TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * (Phi2 * Phi)
    return f, fp, fpp


def _compute_f_upper_map_and_first_two_derivatives(
    x: float, s: float
) -> tuple[float, float, float]:
    f = norm_cdf(-0.5 * s)
    if _is_below_horizon(x):
        fp = -0.5
        fpp = 0
    else:
        w = _square(x / s)
        fp = -0.5 * exp(0.5 * w)
        fpp = SQRT_PI_OVER_TWO * np.exp(w + 0.125 * s * s) * w / s

    return f, fp, fpp


def _square(x: float) -> float:
    return x * x


def _inverse_f_lower_map(x: float, f: float) -> float:
    return (
        0
        if _is_below_horizon(f)
        else fabs(
            x
            / (
                SQRT_THREE
                * inverse_norm_cdf(
                    pow(f / (TWO_PI_OVER_SQRT_TWENTY_SEVEN * fabs(x)), 1.0 / 3.0)
                )
            )
        )
    )


def _inverse_f_upper_map(f: float) -> float:
    return -2.0 * inverse_norm_cdf(f)


def _is_below_horizon(x: float) -> bool:
    """This weeds out denormalized (a.k.a. 'subnormal') numbers."""
    return fabs(x) < DENORMALIZATION_CUTOFF


def _normalized_black_call_using_norm_cdf(x: float, s: float) -> float:
    """
            b(x,s)  =  Φ(x/s+s/2)·exp(x/2)  -   Φ(x/s-s/2)·exp(-x/2)
                =  Φ(h+t)·exp(x/2)      -   Φ(h-t)·exp(-x/2)
    with
                h  =  x/s   and   t  =  s/2
    """
    h = x / s
    t = 0.5 * s
    b_max = exp(0.5 * x)
    b = norm_cdf(h + t) * b_max - norm_cdf(h - t) / b_max
    return fabs(max(b, 0.0))


def _asymptotic_expansion_of_normalized_black_call(h: float, t: float) -> float:
    """
    Asymptotic expansion of

                 b  =  Φ(h+t)·exp(x/2) - Φ(h-t)·exp(-x/2)
    with
                 h  =  x/s   and   t  =  s/2
    which makes
                 b  =  Φ(h+t)·exp(h·t) - Φ(h-t)·exp(-h·t)

                       exp(-(h²+t²)/2)
                    =  ---------------  ·  [ Y(h+t) - Y(h-t) ]
                           √(2π)
    with
              Y(z) := Φ(z)/φ(z)

    for large negative (t-|h|) by the aid of Abramowitz & Stegun (26.2.12) where Φ(z) = φ(z)/|z|·[1-1/z^2+...].
    We define
                        r
            A(h,t) :=  --- · [ Y(h+t) - Y(h-t) ]
                        t

    with r := (h+t)·(h-t) and give an expansion for A(h,t) in q:=(h/r)² expressed in terms of e:=(t/h)² .
    """
    e = (t / h) * (t / h)
    r = (h + t) * (h - t)
    q = (h / r) * (h / r)
    # 17th order asymptotic expansion of A(h,t) in q, sufficient for Φ(h) [and thus y(h)] to have relative accuracy of 1.64E-16 for h <= η  with  η:=-10.
    asymptotic_expansion_sum = 2.0 + q * (
        -6.0e0
        - 2.0 * e
        + 3.0
        * q
        * (
            1.0e1
            + e * (2.0e1 + 2.0 * e)
            + 5.0
            * q
            * (
                -1.4e1
                + e * (-7.0e1 + e * (-4.2e1 - 2.0 * e))
                + 7.0
                * q
                * (
                    1.8e1
                    + e * (1.68e2 + e * (2.52e2 + e * (7.2e1 + 2.0 * e)))
                    + 9.0
                    * q
                    * (
                        -2.2e1
                        + e
                        * (
                            -3.3e2
                            + e * (-9.24e2 + e * (-6.6e2 + e * (-1.1e2 - 2.0 * e)))
                        )
                        + 1.1e1
                        * q
                        * (
                            2.6e1
                            + e
                            * (
                                5.72e2
                                + e
                                * (
                                    2.574e3
                                    + e
                                    * (3.432e3 + e * (1.43e3 + e * (1.56e2 + 2.0 * e)))
                                )
                            )
                            + 1.3e1
                            * q
                            * (
                                -3.0e1
                                + e
                                * (
                                    -9.1e2
                                    + e
                                    * (
                                        -6.006e3
                                        + e
                                        * (
                                            -1.287e4
                                            + e
                                            * (
                                                -1.001e4
                                                + e * (-2.73e3 + e * (-2.1e2 - 2.0 * e))
                                            )
                                        )
                                    )
                                )
                                + 1.5e1
                                * q
                                * (
                                    3.4e1
                                    + e
                                    * (
                                        1.36e3
                                        + e
                                        * (
                                            1.2376e4
                                            + e
                                            * (
                                                3.8896e4
                                                + e
                                                * (
                                                    4.862e4
                                                    + e
                                                    * (
                                                        2.4752e4
                                                        + e
                                                        * (
                                                            4.76e3
                                                            + e * (2.72e2 + 2.0 * e)
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    + 1.7e1
                                    * q
                                    * (
                                        -3.8e1
                                        + e
                                        * (
                                            -1.938e3
                                            + e
                                            * (
                                                -2.3256e4
                                                + e
                                                * (
                                                    -1.00776e5
                                                    + e
                                                    * (
                                                        -1.84756e5
                                                        + e
                                                        * (
                                                            -1.51164e5
                                                            + e
                                                            * (
                                                                -5.4264e4
                                                                + e
                                                                * (
                                                                    -7.752e3
                                                                    + e
                                                                    * (
                                                                        -3.42e2
                                                                        - 2.0 * e
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        + 1.9e1
                                        * q
                                        * (
                                            4.2e1
                                            + e
                                            * (
                                                2.66e3
                                                + e
                                                * (
                                                    4.0698e4
                                                    + e
                                                    * (
                                                        2.3256e5
                                                        + e
                                                        * (
                                                            5.8786e5
                                                            + e
                                                            * (
                                                                7.05432e5
                                                                + e
                                                                * (
                                                                    4.0698e5
                                                                    + e
                                                                    * (
                                                                        1.08528e5
                                                                        + e
                                                                        * (
                                                                            1.197e4
                                                                            + e
                                                                            * (
                                                                                4.2e2
                                                                                + 2.0
                                                                                * e
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                            + 2.1e1
                                            * q
                                            * (
                                                -4.6e1
                                                + e
                                                * (
                                                    -3.542e3
                                                    + e
                                                    * (
                                                        -6.7298e4
                                                        + e
                                                        * (
                                                            -4.90314e5
                                                            + e
                                                            * (
                                                                -1.63438e6
                                                                + e
                                                                * (
                                                                    -2.704156e6
                                                                    + e
                                                                    * (
                                                                        -2.288132e6
                                                                        + e
                                                                        * (
                                                                            -9.80628e5
                                                                            + e
                                                                            * (
                                                                                -2.01894e5
                                                                                + e
                                                                                * (
                                                                                    -1.771e4
                                                                                    + e
                                                                                    * (
                                                                                        -5.06e2
                                                                                        - 2.0
                                                                                        * e
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                                + 2.3e1
                                                * q
                                                * (
                                                    5.0e1
                                                    + e
                                                    * (
                                                        4.6e3
                                                        + e
                                                        * (
                                                            1.0626e5
                                                            + e
                                                            * (
                                                                9.614e5
                                                                + e
                                                                * (
                                                                    4.08595e6
                                                                    + e
                                                                    * (
                                                                        8.9148e6
                                                                        + e
                                                                        * (
                                                                            1.04006e7
                                                                            + e
                                                                            * (
                                                                                6.53752e6
                                                                                + e
                                                                                * (
                                                                                    2.16315e6
                                                                                    + e
                                                                                    * (
                                                                                        3.542e5
                                                                                        + e
                                                                                        * (
                                                                                            2.53e4
                                                                                            + e
                                                                                            * (
                                                                                                6.0e2
                                                                                                + 2.0
                                                                                                * e
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                    + 2.5e1
                                                    * q
                                                    * (
                                                        -5.4e1
                                                        + e
                                                        * (
                                                            -5.85e3
                                                            + e
                                                            * (
                                                                -1.6146e5
                                                                + e
                                                                * (
                                                                    -1.77606e6
                                                                    + e
                                                                    * (
                                                                        -9.37365e6
                                                                        + e
                                                                        * (
                                                                            -2.607579e7
                                                                            + e
                                                                            * (
                                                                                -4.01166e7
                                                                                + e
                                                                                * (
                                                                                    -3.476772e7
                                                                                    + e
                                                                                    * (
                                                                                        -1.687257e7
                                                                                        + e
                                                                                        * (
                                                                                            -4.44015e6
                                                                                            + e
                                                                                            * (
                                                                                                -5.9202e5
                                                                                                + e
                                                                                                * (
                                                                                                    -3.51e4
                                                                                                    + e
                                                                                                    * (
                                                                                                        -7.02e2
                                                                                                        - 2.0
                                                                                                        * e
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        + 2.7e1
                                                        * q
                                                        * (
                                                            5.8e1
                                                            + e
                                                            * (
                                                                7.308e3
                                                                + e
                                                                * (
                                                                    2.3751e5
                                                                    + e
                                                                    * (
                                                                        3.12156e6
                                                                        + e
                                                                        * (
                                                                            2.003001e7
                                                                            + e
                                                                            * (
                                                                                6.919458e7
                                                                                + e
                                                                                * (
                                                                                    1.3572783e8
                                                                                    + e
                                                                                    * (
                                                                                        1.5511752e8
                                                                                        + e
                                                                                        * (
                                                                                            1.0379187e8
                                                                                            + e
                                                                                            * (
                                                                                                4.006002e7
                                                                                                + e
                                                                                                * (
                                                                                                    8.58429e6
                                                                                                    + e
                                                                                                    * (
                                                                                                        9.5004e5
                                                                                                        + e
                                                                                                        * (
                                                                                                            4.7502e4
                                                                                                            + e
                                                                                                            * (
                                                                                                                8.12e2
                                                                                                                + 2.0
                                                                                                                * e
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                            + 2.9e1
                                                            * q
                                                            * (
                                                                -6.2e1
                                                                + e
                                                                * (
                                                                    -8.99e3
                                                                    + e
                                                                    * (
                                                                        -3.39822e5
                                                                        + e
                                                                        * (
                                                                            -5.25915e6
                                                                            + e
                                                                            * (
                                                                                -4.032015e7
                                                                                + e
                                                                                * (
                                                                                    -1.6934463e8
                                                                                    + e
                                                                                    * (
                                                                                        -4.1250615e8
                                                                                        + e
                                                                                        * (
                                                                                            -6.0108039e8
                                                                                            + e
                                                                                            * (
                                                                                                -5.3036505e8
                                                                                                + e
                                                                                                * (
                                                                                                    -2.8224105e8
                                                                                                    + e
                                                                                                    * (
                                                                                                        -8.870433e7
                                                                                                        + e
                                                                                                        * (
                                                                                                            -1.577745e7
                                                                                                            + e
                                                                                                            * (
                                                                                                                -1.472562e6
                                                                                                                + e
                                                                                                                * (
                                                                                                                    -6.293e4
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        -9.3e2
                                                                                                                        - 2.0
                                                                                                                        * e
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                + 3.1e1
                                                                * q
                                                                * (
                                                                    6.6e1
                                                                    + e
                                                                    * (
                                                                        1.0912e4
                                                                        + e
                                                                        * (
                                                                            4.74672e5
                                                                            + e
                                                                            * (
                                                                                8.544096e6
                                                                                + e
                                                                                * (
                                                                                    7.71342e7
                                                                                    + e
                                                                                    * (
                                                                                        3.8707344e8
                                                                                        + e
                                                                                        * (
                                                                                            1.14633288e9
                                                                                            + e
                                                                                            * (
                                                                                                2.07431664e9
                                                                                                + e
                                                                                                * (
                                                                                                    2.33360622e9
                                                                                                    + e
                                                                                                    * (
                                                                                                        1.6376184e9
                                                                                                        + e
                                                                                                        * (
                                                                                                            7.0963464e8
                                                                                                            + e
                                                                                                            * (
                                                                                                                1.8512208e8
                                                                                                                + e
                                                                                                                * (
                                                                                                                    2.7768312e7
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        2.215136e6
                                                                                                                        + e
                                                                                                                        * (
                                                                                                                            8.184e4
                                                                                                                            + e
                                                                                                                            * (
                                                                                                                                1.056e3
                                                                                                                                + 2.0
                                                                                                                                * e
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                    + 3.3e1
                                                                    * (
                                                                        -7.0e1
                                                                        + e
                                                                        * (
                                                                            -1.309e4
                                                                            + e
                                                                            * (
                                                                                -6.49264e5
                                                                                + e
                                                                                * (
                                                                                    -1.344904e7
                                                                                    + e
                                                                                    * (
                                                                                        -1.4121492e8
                                                                                        + e
                                                                                        * (
                                                                                            -8.344518e8
                                                                                            + e
                                                                                            * (
                                                                                                -2.9526756e9
                                                                                                + e
                                                                                                * (
                                                                                                    -6.49588632e9
                                                                                                    + e
                                                                                                    * (
                                                                                                        -9.0751353e9
                                                                                                        + e
                                                                                                        * (
                                                                                                            -8.1198579e9
                                                                                                            + e
                                                                                                            * (
                                                                                                                -4.6399188e9
                                                                                                                + e
                                                                                                                * (
                                                                                                                    -1.6689036e9
                                                                                                                    + e
                                                                                                                    * (
                                                                                                                        -3.67158792e8
                                                                                                                        + e
                                                                                                                        * (
                                                                                                                            -4.707164e7
                                                                                                                            + e
                                                                                                                            * (
                                                                                                                                -3.24632e6
                                                                                                                                + e
                                                                                                                                * (
                                                                                                                                    -1.0472e5
                                                                                                                                    + e
                                                                                                                                    * (
                                                                                                                                        -1.19e3
                                                                                                                                        - 2.0
                                                                                                                                        * e
                                                                                                                                    )
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )
                                                                                                                    )
                                                                                                                )
                                                                                                            )
                                                                                                        )
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                    * q
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    b = (
        ONE_OVER_SQRT_TWO_PI
        * exp((-0.5 * (h * h + t * t)))
        * (t / r)
        * asymptotic_expansion_sum
    )
    return fabs(max(b, 0.0))


def _small_t_expansion_of_normalized_black_call(h, t):
    """
    Calculation of

                 b  =  Φ(h+t)·exp(h·t) - Φ(h-t)·exp(-h·t)

                       exp(-(h²+t²)/2)
                    =  --------------- ·  [ Y(h+t) - Y(h-t) ]
                           √(2π)
    with
              Y(z) := Φ(z)/φ(z)

    using an expansion of Y(h+t)-Y(h-t) for small t to twelvth order in t.
    Theoretically accurate to (better than) precision  ε = 2.23E-16  when  h<=0  and  t < τ  with  τ := 2·ε^(1/16) ≈ 0.21.
    The main bottleneck for precision is the coefficient a:=1+h·Y(h) when |h|>1 .

    :param h:
    :type h: float
    :param t:
    :type t: float

    :return:
    :rtype: float
    """

    # Y(h) := Φ(h)/φ(h) = √(π/2)·erfcx(-h/√2)
    # a := 1+h·Y(h)  --- Note that due to h<0, and h·Y(h) -> -1 (from above) as h -> -∞, we also have that a>0 and a -> 0 as h -> -∞
    # w := t² , h2 := h²
    a = 1 + h * (0.5 * SQRT_TWO_PI) * erfcx_cody(-ONE_OVER_SQRT_TWO * h)
    w = t * t
    h2 = h * h
    expansion = (
        2
        * t
        * (
            a
            + w
            * (
                (-1 + 3 * a + a * h2) / 6
                + w
                * (
                    (-7 + 15 * a + h2 * (-1 + 10 * a + a * h2)) / 120
                    + w
                    * (
                        (
                            -57
                            + 105 * a
                            + h2 * (-18 + 105 * a + h2 * (-1 + 21 * a + a * h2))
                        )
                        / 5040
                        + w
                        * (
                            (
                                -561
                                + 945 * a
                                + h2
                                * (
                                    -285
                                    + 1260 * a
                                    + h2 * (-33 + 378 * a + h2 * (-1 + 36 * a + a * h2))
                                )
                            )
                            / 362880
                            + w
                            * (
                                (
                                    -6555
                                    + 10395 * a
                                    + h2
                                    * (
                                        -4680
                                        + 17325 * a
                                        + h2
                                        * (
                                            -840
                                            + 6930 * a
                                            + h2
                                            * (
                                                -52
                                                + 990 * a
                                                + h2 * (-1 + 55 * a + a * h2)
                                            )
                                        )
                                    )
                                )
                                / 39916800
                                + (
                                    (
                                        -89055
                                        + 135135 * a
                                        + h2
                                        * (
                                            -82845
                                            + 270270 * a
                                            + h2
                                            * (
                                                -20370
                                                + 135135 * a
                                                + h2
                                                * (
                                                    -1926
                                                    + 25740 * a
                                                    + h2
                                                    * (
                                                        -75
                                                        + 2145 * a
                                                        + h2 * (-1 + 78 * a + a * h2)
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    * w
                                )
                                / 6227020800.0
                            )
                        )
                    )
                )
            )
        )
    )
    b = ONE_OVER_SQRT_TWO_PI * exp((-0.5 * (h * h + t * t))) * expansion
    return fabs(max(b, 0.0))


def _normalised_black_call_using_erfcx(h: float, t: float) -> float:
    """
    Given h = x/s and t = s/2, the normalised Black function can be written as

        b(x,s)  =  Φ(x/s+s/2)·exp(x/2)  -   Φ(x/s-s/2)·exp(-x/2)
                =  Φ(h+t)·exp(h·t)      -   Φ(h-t)·exp(-h·t) .                     (*)

    It is mentioned in section 4 (and discussion of figures 2 and 3) of George Marsaglia's article "Evaluating the
    Normal Distribution" (available at http:#www.jstatsoft.org/v11/a05/paper) that the error of any cumulative normal
    function Φ(z) is dominated by the hardware (or compiler implementation) accuracy of exp(-z²/2) which is not
    reliably more than 14 digits when z is large. The accuracy of Φ(z) typically starts coming down to 14 digits when
    z is around -8. For the (normalised) Black function, as above in (*), this means that we are subtracting two terms
    that are each products of terms with about 14 digits of accuracy. The net result, in each of the products, is even
    less accuracy, and then we are taking the difference of these terms, resulting in even less accuracy. When we are
    using the asymptotic expansion asymptotic_expansion_of_normalized_black_call() invoked in the second branch at the
    beginning of this function, we are using only *one* exponential instead of 4, and this improves accuracy. It
    actually improves it a bit more than you would expect from the above logic, namely, almost the full two missing
    digits (in 64 bit IEEE floating point).  Unfortunately, going higher order in the asymptotic expansion will not
    enable us to gain more accuracy (by extending the range in which we could use the expansion) since the asymptotic
    expansion, being a divergent series, can never gain 16 digits of accuracy for z=-8 or just below. The best you can
    get is about 15 digits (just), for about 35 terms in the series (26.2.12), which would result in an prohibitively
    long expression in function asymptotic expansion asymptotic_expansion_of_normalized_black_call(). In this last branch,
    here, we therefore take a different tack as follows.
        The "scaled complementary error function" is defined as erfcx(z) = exp(z²)·erfc(z). Cody's implementation of this
    function as published in "Rational Chebyshev approximations for the error function", W. J. Cody, Math. Comp., 1969, pp.
    631-638, uses rational functions that theoretically approximates erfcx(x) to at least 18 significant decimal digits,
    *without* the use of the exponential function when x>4, which translates to about z<-5.66 in Φ(z). To make use of it,
    we write
                Φ(z) = exp(-z²/2)·erfcx(-z/√2)/2

    to transform the normalised black function to

      b   =  ½ · exp(-½(h²+t²)) · [ erfcx(-(h+t)/√2) -  erfcx(-(h-t)/√2) ]

    which now involves only one exponential, instead of three, when |h|+|t| > 5.66 , and the difference inside the
    square bracket is between the evaluation of two rational functions, which, typically, according to Marsaglia,
    retains the full 16 digits of accuracy (or just a little less than that).
    """

    b = (
        0.5
        * exp(-0.5 * (h * h + t * t))
        * (
            erfcx_cody(-ONE_OVER_SQRT_TWO * (h + t))
            - erfcx_cody(-ONE_OVER_SQRT_TWO * (h - t))
        )
    )
    return fabs(max(b, 0.0))


def _unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    beta: float, x: float, q: float, N: int
) -> float:
    """
    See http://en.wikipedia.org/wiki/Householder%27s_method for a detailed explanation of the third order Householder iteration.

    Given the objective function g(s) whose root x such that 0 = g(s) we seek, iterate

        s_n+1  =  s_n  -  (g/g') · [ 1 - (g''/g')·(g/g') ] / [ 1 - (g/g')·( (g''/g') - (g'''/g')·(g/g')/6 ) ]

    Denoting  newton:=-(g/g'), halley:=(g''/g'), and hh3:=(g'''/g'), this reads

        s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]


    NOTE that this function returns 0 when beta<intrinsic without any safety checks.
    """
    # Subtract intrinsic.
    if q * x > 0:
        beta = fabs(max(beta - _normalised_intrinsic(x, q), 0.0))
        q = -q
    # Map puts to calls
    if q < 0:
        x = -x
        q = -q
    if beta <= 0:  # For negative or zero prices we return 0.
        return 0
    if (
        beta < DENORMALIZATION_CUTOFF
    ):  # For positive but denormalized (a.k.a. 'subnormal') prices, we return 0 since it would be impossible to converge to full machine accuracy anyway.
        return 0
    b_max = exp(0.5 * x)
    if beta >= b_max:
        raise AboveMaximumException
    iterations = 0
    direction_reversal_count = 0
    f = -DBL_MAX
    s = -DBL_MAX
    ds = s
    ds_previous = 0
    s_left = DBL_MIN
    s_right = DBL_MAX
    # The temptation is great to use the optimised form b_c = exp(x/2)/2-exp(-x/2)·Phi(sqrt(-2·x)) but that would require implementing all of the above types of round-off and over/underflow handling for this expression, too.
    s_c = sqrt(fabs(2 * x))
    b_c = normalised_black_call(x, s_c)
    v_c = normalised_vega(x, s_c)
    # Four branches.
    if beta < b_c:
        s_l = s_c - b_c / v_c
        b_l = normalised_black_call(x, s_l)
        if beta < b_l:
            (
                f_lower_map_l,
                d_f_lower_map_l_d_beta,
                d2_f_lower_map_l_d_beta2,
            ) = _compute_f_lower_map_and_first_two_derivatives(x, s_l)
            r_ll = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                0.0,
                b_l,
                0.0,
                f_lower_map_l,
                1.0,
                d_f_lower_map_l_d_beta,
                d2_f_lower_map_l_d_beta2,
                True,
            )
            f = rational_cubic_interpolation(
                beta, 0.0, b_l, 0.0, f_lower_map_l, 1.0, d_f_lower_map_l_d_beta, r_ll
            )
            if not (
                f > 0
            ):  # This can happen due to roundoff truncation for extreme values such as |x|>500.
                # We switch to quadratic interpolation using f(0)≡0, f(b_l), and f'(0)≡1 to specify the quadratic.
                t = beta / b_l
                f = (f_lower_map_l * t + b_l * (1 - t)) * t

            s = _inverse_f_lower_map(x, f)
            s_right = s_l
            #
            # In this branch, which comprises the lowest segment, the objective function is
            #     g(s) = 1/ln(b(x,s)) - 1/ln(beta)
            #          ≡ 1/ln(b(s)) - 1/ln(beta)
            # This makes
            #              g'               =   -b'/(b·ln(b)²)
            #              newton = -g/g'   =   (ln(beta)-ln(b))·ln(b)/ln(beta)·b/b'
            #              halley = g''/g'  =   b''/b'  -  b'/b·(1+2/ln(b))
            #              hh3    = g'''/g' =   b'''/b' +  2(b'/b)²·(1+3/ln(b)·(1+1/ln(b)))  -  3(b''/b)·(1+2/ln(b))
            #
            # The Householder(3) iteration is
            #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
            #
            while iterations < N and fabs(ds) > DBL_EPSILON * s:
                if ds * ds_previous < 0:
                    direction_reversal_count += 1
                if iterations > 0 and (
                    3 == direction_reversal_count or not (s > s_left and s < s_right)
                ):
                    # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                    # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                    s = 0.5 * (s_left + s_right)
                    if s_right - s_left <= DBL_EPSILON * s:
                        break
                    direction_reversal_count = 0
                    ds = 0
                ds_previous = ds
                b = normalised_black_call(x, s)
                bp = normalised_vega(x, s)
                if b > beta and s < s_right:
                    s_right = s
                elif b < beta and s > s_left:
                    s_left = s  # Tighten the bracket if applicable.
                if (
                    b <= 0 or bp <= 0
                ):  # Numerical underflow. Switch to binary nesting for this iteration.
                    ds = 0.5 * (s_left + s_right) - s
                else:
                    ln_b = log(b)
                    ln_beta = log(beta)
                    bpob = bp / b
                    h = x / s
                    b_halley = h * h / s - s / 4
                    newton = (ln_beta - ln_b) * ln_b / ln_beta / bpob
                    halley = b_halley - bpob * (1 + 2 / ln_b)
                    b_hh3 = b_halley * b_halley - 3 * _square(h / s) - 0.25
                    hh3 = (
                        b_hh3
                        + 2 * _square(bpob) * (1 + 3 / ln_b * (1 + 1 / ln_b))
                        - 3 * b_halley * bpob * (1 + 2 / ln_b)
                    )
                    ds = newton * _householder_factor(newton, halley, hh3)
                ds = max(-0.5 * s, ds)
                s += ds
                iterations += 1
            return s
        else:
            v_l = normalised_vega(x, s_l)
            r_lm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, 0.0, False
            )
            s = rational_cubic_interpolation(
                beta, b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, r_lm
            )
            s_left = s_l
            s_right = s_c
    else:
        s_h = s_c + (b_max - b_c) / v_c if v_c > DBL_MIN else s_c
        b_h = normalised_black_call(x, s_h)
        if beta <= b_h:
            v_h = normalised_vega(x, s_h)
            r_hm = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                b_c, b_h, s_c, s_h, 1 / v_c, 1 / v_h, 0.0, False
            )
            s = rational_cubic_interpolation(
                beta, b_c, b_h, s_c, s_h, 1 / v_c, 1 / v_h, r_hm
            )
            s_left = s_c
            s_right = s_h
        else:
            (
                f_upper_map_h,
                d_f_upper_map_h_d_beta,
                d2_f_upper_map_h_d_beta2,
            ) = _compute_f_upper_map_and_first_two_derivatives(x, s_h)
            if d2_f_upper_map_h_d_beta2 > -SQRT_DBL_MAX < SQRT_DBL_MAX:
                r_hh = convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                    b_h,
                    b_max,
                    f_upper_map_h,
                    0.0,
                    d_f_upper_map_h_d_beta,
                    -0.5,
                    d2_f_upper_map_h_d_beta2,
                    True,
                )
                f = rational_cubic_interpolation(
                    beta,
                    b_h,
                    b_max,
                    f_upper_map_h,
                    0.0,
                    d_f_upper_map_h_d_beta,
                    -0.5,
                    r_hh,
                )
            if f <= 0:
                h = b_max - b_h
                t = (beta - b_h) / h
                f = (f_upper_map_h * (1 - t) + 0.5 * h * t) * (
                    1 - t
                )  # We switch to quadratic interpolation using f(b_h), f(b_max)≡0, and f'(b_max)≡-1/2 to specify the quadratic.
            s = _inverse_f_upper_map(f)
            s_left = s_h
            if (
                beta > 0.5 * b_max
            ):  # Else we better drop through and let the objective function be g(s) = b(x,s)-beta.
                #
                # In this branch, which comprises the upper segment, the objective function is
                #     g(s) = ln(b_max-beta)-ln(b_max-b(x,s))
                #          ≡ ln((b_max-beta)/(b_max-b(s)))
                # This makes
                #              g'               =   b'/(b_max-b)
                #              newton = -g/g'   =   ln((b_max-b)/(b_max-beta))·(b_max-b)/b'
                #              halley = g''/g'  =   b''/b'  +  b'/(b_max-b)
                #              hh3    = g'''/g' =   b'''/b' +  g'·(2g'+3b''/b')
                # and the iteration is
                #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
                #
                while iterations < N and fabs(ds) > DBL_EPSILON * s:
                    if ds * ds_previous < 0:
                        direction_reversal_count += 1
                    if iterations > 0 and (
                        3 == direction_reversal_count
                        or not (s > s_left and s < s_right)
                    ):
                        # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                        # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                        s = 0.5 * (s_left + s_right)
                    if s_right - s_left <= DBL_EPSILON * s:
                        break
                    direction_reversal_count = 0
                    ds = 0
                    ds_previous = ds
                    b = normalised_black_call(x, s)
                    bp = normalised_vega(x, s)
                    if b > beta and s < s_right:
                        s_right = s
                    elif b < beta and s > s_left:
                        s_left = s  # Tighten the bracket if applicable.
                    if (
                        b >= b_max or bp <= DBL_MIN
                    ):  # Numerical underflow. Switch to binary nesting for this iteration.
                        ds = 0.5 * (s_left + s_right) - s
                    else:
                        b_max_minus_b = b_max - b
                        g = log((b_max - beta) / b_max_minus_b)
                        gp = bp / b_max_minus_b
                        b_halley = _square(x / s) / s - s / 4
                        b_hh3 = b_halley * b_halley - 3 * _square(x / (s * s)) - 0.25
                        newton = -g / gp
                        halley = b_halley + gp
                        hh3 = b_hh3 + gp * (2 * gp + 3 * b_halley)
                        ds = newton * _householder_factor(newton, halley, hh3)
                    ds = max(-0.5 * s, ds)
                    s += ds
                    iterations += 1
                return s
    # In this branch, which comprises the two middle segments, the objective function is g(s) = b(x,s)-beta, or g(s) = b(s) - beta, for short.
    # This makes
    #              newton = -g/g'   =  -(b-beta)/b'
    #              halley = g''/g'  =    b''/b'    =  x²/s³-s/4
    #              hh3    = g'''/g' =    b'''/b'   =  halley² - 3·(x/s²)² - 1/4
    # and the iteration is
    #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
    #
    while iterations < N and fabs(ds) > DBL_EPSILON * s:
        if ds * ds_previous < 0:
            direction_reversal_count += 1
        if iterations > 0 and (
            3 == direction_reversal_count or not (s > s_left and s < s_right)
        ):
            # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
            # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
            s = 0.5 * (s_left + s_right)
            if s_right - s_left <= DBL_EPSILON * s:
                break
            direction_reversal_count = 0
            ds = 0
        ds_previous = ds
        b = normalised_black_call(x, s)
        bp = normalised_vega(x, s)
        if b > beta and s < s_right:
            s_right = s
        elif b < beta and s > s_left:
            s_left = s  # Tighten the bracket if applicable.
        newton = (beta - b) / bp
        halley = _square(x / s) / s - s / 4
        hh3 = halley * halley - 3 * _square(x / (s * s)) - 0.25
        ds = max(-0.5 * s, newton * _householder_factor(newton, halley, hh3))
        s += ds
        iterations += 1
    return s


def normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    beta: float, x: float, q: float, N: int
) -> float:
    # Map in-the-money to out-of-the-money
    if q * x > 0:
        beta -= _normalised_intrinsic(x, q)
        q = -q

    if beta < 0:
        raise BelowIntrinsicException
    return _unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        beta, x, q, N
    )


def implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
    price: float, F: float, K: float, T: float, q: float, N: int
) -> float:
    intrinsic = fabs(max(K - F if q < 0 else F - K, 0.0))
    if price < intrinsic:
        raise BelowIntrinsicException
    max_price = K if q < 0 else F
    if price >= max_price:
        raise AboveMaximumException
    x = log(F / K)
    # Map in-the-money to out-of-the-money
    if q * x > 0:
        price = fabs(max(price - intrinsic, 0.0))
        q = -q
    return _unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        price / (sqrt(F) * sqrt(K)), x, q, N
    ) / sqrt(
        T
    )


def normalised_implied_volatility_from_a_transformed_rational_guess(
    beta: float, x: float, q: float
) -> float:
    return normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        beta, x, q, implied_volatility_maximum_iterations
    )


def implied_volatility_from_a_transformed_rational_guess(
    price: float, F: float, K: float, T: float, q: float
) -> float:
    return implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        price, F, K, T, q, implied_volatility_maximum_iterations
    )


def normalised_vega(x: float, s: float) -> float:
    ax = fabs(x)
    if ax <= 0:
        return ONE_OVER_SQRT_TWO_PI * exp(-0.125 * s * s)
    else:
        return (
            0
            if s <= 0 or s <= ax * SQRT_DBL_MIN
            else ONE_OVER_SQRT_TWO_PI * exp(-0.5 * (_square(x / s) + _square(0.5 * s)))
        )


def _normalised_intrinsic(x: float, q: float) -> float:
    if q * x <= 0:
        return 0
    x2 = x * x
    if (
        x2 < 98 * FOURTH_ROOT_DBL_EPSILON
    ):  # The factor 98 is computed from last coefficient: √√92897280 = 98.1749
        return fabs(
            max(
                (-1 if q < 0 else 1)
                * x
                * (
                    1
                    + x2
                    * (
                        (1.0 / 24.0)
                        + x2
                        * (
                            (1.0 / 1920.0)
                            + x2 * ((1.0 / 322560.0) + (1.0 / 92897280.0) * x2)
                        )
                    )
                ),
                0.0,
            )
        )
    b_max = exp(0.5 * x)
    one_over_b_max = 1 / b_max
    return fabs(max((-1 if q < 0 else 1) * (b_max - one_over_b_max), 0.0))


def _normalised_intrinsic_call(x: float) -> float:
    return _normalised_intrinsic(x, 1)


def normalised_black_call(x: float, s: float) -> float:
    if x > 0:
        return _normalised_intrinsic_call(x) + normalised_black_call(-x, s)
    ax = fabs(x)
    if s <= ax * DENORMALIZATION_CUTOFF:
        return _normalised_intrinsic_call(x)
    # Denote h := x/s and t := s/2. We evaluate the condition |h|>|η|, i.e., h<η  &&  t < τ+|h|-|η|  avoiding any
    # divisions by s , where η = asymptotic_expansion_accuracy_threshold  and τ =
    # small_t_expansion_of_normalized_black_threshold .
    if x < s * asymptotic_expansion_accuracy_threshold and 0.5 * s * s + x < s * (
        small_t_expansion_of_normalized_black_threshold
        + asymptotic_expansion_accuracy_threshold
    ):
        # Region 1.
        return _asymptotic_expansion_of_normalized_black_call(x / s, 0.5 * s)
    if 0.5 * s < small_t_expansion_of_normalized_black_threshold:
        # Region 2.
        return _small_t_expansion_of_normalized_black_call(x / s, 0.5 * s)
    # When b is more than, say, about 85% of b_max=exp(x/2), then b is dominated by the first of the two terms in the
    #  Black formula, and we retain more accuracy by not attempting to combine the two terms in any way. We evaluate
    # the condition h+t>0.85  avoiding any divisions by s.
    if x + 0.5 * s * s > s * 0.85:
        # Region 3.
        return _normalized_black_call_using_norm_cdf(x, s)
    # Region 4.
    return _normalised_black_call_using_erfcx(x / s, 0.5 * s)


def normalised_black(x: float, s: float, q: float) -> float:
    """q = ±1"""
    return normalised_black_call(
        -x if q < 0 else x, s
    )  # Reciprocal-strike call-put equivalence


def black(F: float, K: float, sigma: float, T: float, q: float) -> float:
    intrinsic = fabs(max((K - F if q < 0 else F - K), 0.0))
    # Map in-the-money to out-of-the-money
    if q * (F - K) > 0:
        return intrinsic + black(F, K, sigma, T, -q)
    return max(
        intrinsic,
        (sqrt(F) * sqrt(K)) * normalised_black(log(F / K), sigma * sqrt(T), q),
    )
