import unittest
from math import log, sqrt

import pylbr


class TestPublicFunctions(unittest.TestCase):
    def _assertAlmostEqual(self, actual, expected, epsilon=1.0e-12):
        if actual is None or expected is None:
            self.fail("{} != {}".format(actual, expected))
        self.assertTrue(
            abs(actual - expected) < epsilon, "{} != {}".format(actual, expected)
        )

    def test_black(self):
        F = 100
        K = 100
        sigma = 0.2
        T = 0.5
        q = 1  # CALL = 1 PUT = -1

        actual = pylbr.black(F, K, sigma, T, q)
        expected = 5.637197779701664
        self._assertAlmostEqual(actual, expected)

    def test_implied_volatility_from_a_transformed_rational_guess(self):
        F = 100
        K = 100
        sigma = 0.2
        T = 0.5
        q = 1  # CALL = 1 PUT = -1

        price = 5.637197779701664
        actual = pylbr.implied_volatility_from_a_transformed_rational_guess(
            price, F, K, T, q
        )
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

    def test_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        self,
    ):
        F = 100
        K = 100
        sigma = 0.232323232
        T = 0.5
        q = 1  # CALL = 1 PUT = -1
        N = 1

        price = 6.54635543387
        actual = pylbr.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price, F, K, T, q, N
        )
        expected = 0.232323232
        self._assertAlmostEqual(actual, expected)

    def test_normalised_black(self):
        F = 100
        K = 95
        T = 0.5
        sigma = 0.3

        x = log(F / K)
        s = sigma * sqrt(T)

        q = -1  # CALL = 1 PUT = -1
        actual_put = pylbr.normalised_black(x, s, q)
        expected_put = 0.061296663817558904
        self._assertAlmostEqual(actual_put, expected_put)

        q = 1  # CALL = 1 PUT = -1
        actual_call = pylbr.normalised_black(x, s, q)
        expected_call = 0.11259558142181655
        self._assertAlmostEqual(actual_call, expected_call)

    def test_normalised_black_call(self):
        F = 100
        K = 95
        T = 0.5
        sigma = 0.3

        x = log(F / K)
        s = sigma * sqrt(T)

        actual = pylbr.normalised_black_call(x, s)
        expected = 0.11259558142181655
        self._assertAlmostEqual(actual, expected)

    def test_normalised_vega(self):
        x = 0.0
        s = 0.0
        actual = pylbr.normalised_vega(x, s)
        expected = 0.3989422804014327
        self._assertAlmostEqual(actual, expected)

        x = 0.0
        s = 2.937528694999807
        actual = pylbr.normalised_vega(x, s)
        expected = 0.13566415614561067
        self._assertAlmostEqual(actual, expected)

        x = 0.0
        s = 0.2
        actual = pylbr.normalised_vega(x, s)
        expected = 0.3969525474770118
        self._assertAlmostEqual(actual, expected)

    def test_normalised_implied_volatility_from_a_transformed_rational_guess(self):
        x = 0.0
        s = 0.2
        q = 1  # CALL = 1 PUT = -1
        beta_call = pylbr.normalised_black(x, s, q)
        actual = pylbr.normalised_implied_volatility_from_a_transformed_rational_guess(
            beta_call, x, q
        )
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

        x = 0.1
        s = 0.23232323888
        q = -1  # CALL = 1 PUT = -1
        beta_put = pylbr.normalised_black(x, s, q)
        actual = pylbr.normalised_implied_volatility_from_a_transformed_rational_guess(
            beta_put, x, q
        )
        expected = 0.23232323888
        self._assertAlmostEqual(actual, expected)

    def test_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        self,
    ):
        x = 0.0
        s = 0.2
        q = 1  # CALL = 1 PUT = -1
        N = 1
        beta_call = pylbr.normalised_black(x, s, q)
        actual = pylbr.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta_call, x, q, N
        )
        expected = 0.2
        self._assertAlmostEqual(actual, expected)

        x = 0.1
        s = 0.23232323888
        q = -1  # CALL = 1 PUT = -1
        N = 1
        beta_put = pylbr.normalised_black(x, s, q)
        actual = pylbr.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            beta_put, x, q, N
        )
        expected = 0.23232323888
        self._assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
