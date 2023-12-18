# -*- coding: utf-8 -*-

"""
py_lets_be_rational
~~~~~~~~~~~~~~~~~~~

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

from pylbr.lets_be_rational import (
    black,
    implied_volatility_from_a_transformed_rational_guess,
    implied_volatility_from_a_transformed_rational_guess_with_limited_iterations,
    normalised_black,
    normalised_black_call,
    normalised_implied_volatility_from_a_transformed_rational_guess,
    normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations,
    normalised_vega,
)
