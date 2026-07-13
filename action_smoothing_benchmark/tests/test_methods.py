import unittest

import numpy as np

from action_smoothing_benchmark.methods import build_method_catalog


class SmoothingMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.methods = {method.method_id: method for method in build_method_catalog()}

    def test_catalog_contains_expected_variants(self) -> None:
        self.assertEqual(len(self.methods), 31)
        for degree in range(1, 6):
            self.assertIn(f"polynomial_{degree}", self.methods)

    def test_polynomial_methods_reconstruct_matching_degree(self) -> None:
        time = np.linspace(-1.0, 1.0, 50)
        for degree in range(1, 6):
            columns = [sum((dimension + 1) * (power + 0.25) * time**power for power in range(degree + 1)) for dimension in range(3)]
            action = np.stack(columns, axis=-1)
            smoothed, _runtime, error = self.methods[f"polynomial_{degree}"].apply(action, 60.0)
            self.assertIsNone(error)
            np.testing.assert_allclose(smoothed, action, atol=1e-10, rtol=1e-10)

    def test_cubic_matches_lerobot_design_matrix_definition(self) -> None:
        rng = np.random.default_rng(7)
        action = rng.normal(size=(50, 12))
        time = np.linspace(-1.0, 1.0, 50)
        design = np.stack([time**power for power in range(4)], axis=-1)
        expected = design @ np.linalg.lstsq(design, action, rcond=None)[0]
        actual, _runtime, error = self.methods["polynomial_3"].apply(action, 60.0)
        self.assertIsNone(error)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_all_methods_preserve_shape_and_finite_values(self) -> None:
        rng = np.random.default_rng(19)
        action = np.cumsum(rng.normal(scale=0.1, size=(50, 12)), axis=0)
        for method in self.methods.values():
            with self.subTest(method=method.method_id):
                output, runtime_ms, error = method.apply(action, 60.0)
                self.assertIsNone(error)
                self.assertEqual(output.shape, action.shape)
                self.assertTrue(np.isfinite(output).all())
                self.assertGreaterEqual(runtime_ms, 0.0)

    def test_short_chunks_fall_back_without_shape_change(self) -> None:
        action = np.arange(36, dtype=np.float64).reshape(3, 12)
        output, _runtime, error = self.methods["polynomial_5"].apply(action, 60.0)
        self.assertIsNotNone(error)
        np.testing.assert_array_equal(output, action)


if __name__ == "__main__":
    unittest.main()

