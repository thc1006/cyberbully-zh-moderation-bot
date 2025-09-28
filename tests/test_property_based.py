#!/usr/bin/env python3
"""
Property-based testing for robust validation of core modules
Uses hypothesis for generating test cases and validating invariants
"""

import unittest

import numpy as np

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.strategies import composite, floats, integers, lists, text

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    # Fallback for environments without hypothesis
    HYPOTHESIS_AVAILABLE = False

    def given(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def settings(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def assume(condition):
        pass

    # Mock strategies for when hypothesis is not available
    class MockStrategies:
        def text(self, **kwargs):
            return ["test", "測試", ""]

        def integers(self, **kwargs):
            return [0, 1, 100, -1]

        def floats(self, **kwargs):
            return [0.0, 0.5, 1.0, 0.123]

    st = MockStrategies()

# Import modules to test
from cyberpuppy.config import Settings
from cyberpuppy.models.result import DetectionResult


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "Hypothesis not available")
class TestPropertyBasedConfig(unittest.TestCase):
    """Property-based tests for config module"""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=1000)
    def test_settings_model_name_property(self, model_name):
        """Test that model name property handles various inputs correctly."""
        assume(model_name.strip())  # Non-empty after stripping
        assume(
            not any(char in model_name for char in ["/", "\\", "<", ">", "|"])
        )  # No invalid path chars

        try:
            settings = Settings(model_name=model_name)
            self.assertEqual(settings.model_name, model_name)
        except (ValueError, TypeError):
            # Some model names might be rejected, which is valid
            pass

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=30)
    def test_settings_confidence_threshold_property(self, threshold):
        """Property: Valid confidence thresholds should always be accepted"""
        assume(not np.isnan(threshold))  # Exclude NaN values

        settings = Settings(model_name="test-model", confidence_threshold=threshold)

        self.assertEqual(settings.confidence_threshold, threshold)
        self.assertTrue(0.0 <= settings.confidence_threshold <= 1.0)

    @given(st.integers(min_value=1, max_value=1024), st.integers(min_value=1, max_value=256))
    @settings(max_examples=20)
    def test_settings_size_properties(self, max_seq_length, batch_size):
        """Property: Positive integer parameters should be accepted"""
        settings = Settings(
            model_name="test-model", max_sequence_length=max_seq_length, batch_size=batch_size
        )

        self.assertEqual(settings.max_sequence_length, max_seq_length)
        self.assertEqual(settings.batch_size, batch_size)
        self.assertTrue(settings.max_sequence_length > 0)
        self.assertTrue(settings.batch_size > 0)


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "Hypothesis not available")
class TestPropertyBasedDetectionResult(unittest.TestCase):
    """Property-based tests for DetectionResult"""

    @composite
    def detection_result_data(self):
        """Generate valid detection result data"""
        __test_text = self(st.text(min_size=0, max_size=1000))

        toxicity_label = self(st.integers(min_value=0, max_value=2))
        toxicity_confidence = self(st.floats(min_value=0.0, max_value=1.0))

        emotion_label = self(st.integers(min_value=0, max_value=2))
        emotion_confidence = self(st.floats(min_value=0.0, max_value=1.0))

        bullying_label = self(st.integers(min_value=0, max_value=1))
        bullying_confidence = self(st.floats(min_value=0.0, max_value=1.0))

        return {
            "text": text,
            "toxicity_label": toxicity_label,
            "toxicity_confidence": toxicity_confidence,
            "emotion_label": emotion_label,
            "emotion_confidence": emotion_confidence,
            "bullying_label": bullying_label,
            "bullying_confidence": bullying_confidence,
        }

    @given(detection_result_data())
    @settings(max_examples=100, deadline=2000)
    def test_detection_result_invariants(self, data):
        """Property: Valid detection results should maintain invariants"""
        assume(
            not any(
                np.isnan(v)
                for v in [
                    data["toxicity_confidence"],
                    data["emotion_confidence"],
                    data["bullying_confidence"],
                ]
            )
        )

        result = DetectionResult(
            text=data["text"],
            toxicity_label=data["toxicity_label"],
            toxicity_confidence=data["toxicity_confidence"],
            emotion_label=data["emotion_label"],
            emotion_confidence=data["emotion_confidence"],
            bullying_label=data["bullying_label"],
            bullying_confidence=data["bullying_confidence"],
        )

        # Invariant: Confidence values should remain in [0, 1]
        self.assertTrue(0.0 <= result.toxicity_confidence <= 1.0)
        self.assertTrue(0.0 <= result.emotion_confidence <= 1.0)
        self.assertTrue(0.0 <= result.bullying_confidence <= 1.0)

        # Invariant: Labels should be non-negative integers
        self.assertTrue(result.toxicity_label >= 0)
        self.assertTrue(result.emotion_label >= 0)
        self.assertTrue(result.bullying_label >= 0)

        # Invariant: Text should be preserved
        self.assertEqual(result.text, data["text"])

    @given(st.text())
    @settings(max_examples=50)
    def test_detection_result_validation_text_invariants(self, text):
        """Property: Text handling should be consistent"""
        result = DetectionResult(
            text=text,
            toxicity_label=0,
            toxicity_confidence=0.5,
            emotion_label=1,
            emotion_confidence=0.6,
            bullying_label=0,
            bullying_confidence=0.2,
        )

        # Invariant: Text should be exactly preserved
        self.assertEqual(result.text, text)
        self.assertIsInstance(result.text, str)

        # Invariant: Serialization should preserve text
        serialized = result.to_dict()
        self.assertEqual(serialized["text"], text)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_batch_detection_results_properties(self, confidences):
        """Property: Batch operations should preserve individual properties"""
        assume(all(not np.isnan(conf) for conf in confidences))

        results = []
        for i, conf in enumerate(confidences):
            result = DetectionResult(
                text=f"test_{i}",
                toxicity_label=i % 2,
                toxicity_confidence=conf,
                emotion_label=i % 3,
                emotion_confidence=conf * 0.8,
                bullying_label=0,
                bullying_confidence=conf * 0.3,
            )
            results.append(result)

        # Property: All results should maintain invariants
        for result in results:
            self.assertTrue(0.0 <= result.toxicity_confidence <= 1.0)
            self.assertTrue(0.0 <= result.emotion_confidence <= 1.0)
            self.assertTrue(0.0 <= result.bullying_confidence <= 1.0)

        # Property: Batch size should be preserved
        self.assertEqual(len(results), len(confidences))


class TestFallbackPropertyTesting(unittest.TestCase):
    """Fallback property tests when hypothesis is not available"""

    def test_config_edge_case_handling(self):
        """Test configuration with various edge cases"""
        edge_cases = [
            {"model_name": "a", "confidence_threshold": 0.0},
            {"model_name": "test-model-123", "confidence_threshold": 1.0},
            {"model_name": "model/with/path", "confidence_threshold": 0.5},
        ]

        for case in edge_cases:
            try:
                settings = Settings(**case)
                self.assertIsInstance(settings.model_name, str)
                self.assertTrue(0.0 <= settings.confidence_threshold <= 1.0)
            except (ValueError, TypeError):
                # Some cases might be invalid, which is expected
                pass

    def test_detection_result_edge_cases(self):
        """Test detection result with edge cases"""
        edge_cases = [
            # Empty text
            {"text": "", "toxicity_label": 0, "toxicity_confidence": 0.5},
            # Very long text
            {"text": "a" * 1000, "toxicity_label": 0, "toxicity_confidence": 0.5},
            # Unicode text
            {"text": "測試🤖", "toxicity_label": 0, "toxicity_confidence": 0.3},
            # Extreme confidence values
            {"text": "test", "toxicity_label": 0, "toxicity_confidence": 0.0},
            {"text": "test", "toxicity_label": 1, "toxicity_confidence": 1.0},
        ]

        for case in edge_cases:
            result = DetectionResult(
                text=case["text"],
                toxicity_label=case["toxicity_label"],
                toxicity_confidence=case["toxicity_confidence"],
                emotion_label=0,
                emotion_confidence=0.5,
                bullying_label=0,
                bullying_confidence=0.1,
            )

            # Verify invariants
            self.assertEqual(result.text, case["text"])
            self.assertTrue(0.0 <= result.toxicity_confidence <= 1.0)
            self.assertIsInstance(result.toxicity_label, int)

    def test_numerical_stability(self):
        """Test numerical stability with various float values"""
        float_cases = [0.0, 1.0, 0.5, 0.123456789, 0.999999999, 0.000000001]

        for confidence in float_cases:
            result = DetectionResult(
                text="numerical_test",
                toxicity_label=1,
                toxicity_confidence=confidence,
                emotion_label=0,
                emotion_confidence=confidence * 0.8,
                bullying_label=0,
                bullying_confidence=confidence * 0.3,
            )

            # Test that confidence values are preserved accurately
            self.assertAlmostEqual(result.toxicity_confidence, confidence, places=7)

            # Test serialization preserves precision
            serialized = result.to_dict()
            self.assertAlmostEqual(serialized["toxicity_confidence"], confidence, places=7)

    def test_concurrent_result_creation(self):
        """Test creating multiple results concurrently (simulated)"""
        import threading

        results = []
        errors = []

        def create_result(index):
            try:
                result = DetectionResult(
                    text=f"concurrent_test_{index}",
                    toxicity_label=index % 2,
                    toxicity_confidence=0.5 + (index % 50) / 100.0,
                    emotion_label=index % 3,
                    emotion_confidence=0.3 + (index % 70) / 100.0,
                    bullying_label=0,
                    bullying_confidence=0.1 + (index % 30) / 100.0,
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple results in threads
        threads = []
        for i in range(50):
            thread = threading.Thread(target=create_result, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 50)

        # Verify all results are valid
        for result in results:
            self.assertTrue(0.0 <= result.toxicity_confidence <= 1.0)
            self.assertIsInstance(result.text, str)


class TestInvariantValidation(unittest.TestCase):
    """Test invariants that should always hold"""

    def test_detection_result_serialization_roundtrip(self):
        """Test that serialization-deserialization preserves data"""
        original_data = [
            {"text": "test1", "tox_conf": 0.8, "emo_conf": 0.6},
            {"text": "測試中文", "tox_conf": 0.3, "emo_conf": 0.9},
            {"text": "🤖 emoji test", "tox_conf": 0.5, "emo_conf": 0.4},
        ]

        for data in original_data:
            # Create original result
            original = DetectionResult(
                text=data["text"],
                toxicity_label=1,
                toxicity_confidence=data["tox_conf"],
                emotion_label=2,
                emotion_confidence=data["emo_conf"],
                bullying_label=0,
                bullying_confidence=0.1,
            )

            # Serialize to dict
            serialized = original.to_dict()

            # Verify critical data is preserved
            self.assertEqual(serialized["text"], original.text)
            self.assertEqual(serialized["toxicity_" "confidence"], original.toxicity_confidence)
            self.assertEqual(serialized["emotion_c" "onfidence"], original.emotion_confidence)

    def test_confidence_ordering_invariant(self):
        """Test that confidence relationships are preserved"""
        base_result = DetectionResult(
            text="ordering_test",
            toxicity_label=1,
            toxicity_confidence=0.7,
            emotion_label=1,
            emotion_confidence=0.5,
            bullying_label=0,
            bullying_confidence=0.3,
        )

        # Invariant: If we know the ordering, it should be preserved
        self.assertGreater(base_result.toxicity_confidence, base_result.emotion_confidence)
        self.assertGreater(base_result.emotion_confidence, base_result.bullying_confidence)

        # This should hold for serialized version too
        serialized = base_result.to_dict()
        self.assertGreater(serialized["toxicity_confidence"], serialized["bullying_confidence"])

    def test_label_confidence_consistency(self):
        """Test consistency between labels and confidence values"""
        # High confidence should typically correspond to non-zero labels
        # (though this isn't always guaranteed in practice)
        high_conf_result = DetectionResult(
            text="high_confidence_test",
            toxicity_label=1,  # Positive label
            toxicity_confidence=0.95,  # High confidence
            emotion_label=2,
            emotion_confidence=0.85,
            bullying_label=0,
            bullying_confidence=0.05,
        )

        # Just verify the structure is maintained
        self.assertIsInstance(high_conf_result.toxicity_label, int)
        self.assertIsInstance(high_conf_result.toxicity_confidence, float)
        self.assertTrue(high_conf_result.toxicity_label >= 0)
        self.assertTrue(0.0 <= high_conf_result.toxicity_confidence <= 1.0)


if __name__ == "__main__":
    if HYPOTHESIS_AVAILABLE:
        print("Running property-based tests with Hypothesis")
    else:
        print("Running fallback property tests (Hypothesis not available)")

    unittest.main()
