"""
CyberPuppy API Integration Test Script

Tests the integrated API with real Chinese toxicity examples to verify:
1. Model loading and inference
2. API response format compliance
3. Performance benchmarks
4. Privacy-compliant logging
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import aiohttp

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TEXTS = [
    # Benign examples
    {
        "text": "你好，今天天氣真好",
        "expected_toxicity": "none",
        "expected_emotion": "pos",
        "description": "Positive greeting",
    },
    {
        "text": "我需要幫助解決這個問題",
        "expected_toxicity": "none",
        "expected_emotion": "neu",
        "description": "Neutral request for help",
    },
    {
        "text": "謝謝你的幫助，我很感激",
        "expected_toxicity": "none",
        "expected_emotion": "pos",
        "description": "Positive gratitude",
    },
    # Toxic examples
    {
        "text": "你真是個笨蛋",
        "expected_toxicity": "toxic",
        "expected_emotion": "neg",
        "description": "Mild insult - toxic",
    },
    {
        "text": "滾開，白痴",
        "expected_toxicity": "toxic",
        "expected_emotion": "neg",
        "description": "Offensive language - toxic",
    },
    {
        "text": "你這個廢物，沒用的東西",
        "expected_toxicity": "toxic",
        "expected_emotion": "neg",
        "description": "Derogatory language - toxic",
    },
    # Severe toxic examples
    {
        "text": "去死吧，我要殺死你",
        "expected_toxicity": "severe",
        "expected_emotion": "neg",
        "description": "Death threats - severe",
    },
    {
        "text": "你不如自殺算了",
        "expected_toxicity": "severe",
        "expected_emotion": "neg",
        "description": "Suicide suggestion - severe",
    },
    # Bullying examples
    {
        "text": "沒有人喜歡你，你很討厭",
        "expected_toxicity": "toxic",
        "expected_bullying": "harassment",
        "expected_emotion": "neg",
        "description": "Social harassment",
    },
    {
        "text": "如果你不聽話，你會後悔的",
        "expected_toxicity": "toxic",
        "expected_bullying": "threat",
        "expected_emotion": "neg",
        "description": "Intimidation threat",
    },
    # Edge cases
    {
        "text": "呵呵呵呵呵",
        "expected_toxicity": "none",
        "expected_emotion": "neu",
        "description": "Ambiguous expression",
    },
    {
        "text": "我今天心情不好",
        "expected_toxicity": "none",
        "expected_emotion": "neg",
        "description": "Negative emotion but not toxic",
    },
    # Mixed content
    {
        "text": "你好，但是你真的很笨",
        "expected_toxicity": "toxic",
        "expected_emotion": "neg",
        "description": "Mixed polite + insult",
    },
]


class APITester:
    """API integration tester with performance benchmarking."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None
        self.test_results = []

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def wait_for_api_startup(self, timeout: int = 60) -> bool:
        """Wait for API to be ready."""
        logger.info("Waiting for API to start up...")

        for attempt in range(timeout):
            try:
                async with self.session.get(f"{self.base_url}/healthz") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        if health_data.get("status") == "healthy":
                            logger.info(f"API ready after {attempt + 1} seconds")
                            return True
                        elif health_data.get("status") == "starting":
                            logger.info(f"API starting... (attempt {attempt + 1})")
                        else:
                            logger.warning(f"API status: {health_data.get('status')}")
            except Exception as e:
                logger.debug(f"Startup check failed (attempt {attempt + 1}): {e}")

            await asyncio.sleep(1)

        logger.error("API failed to start within timeout")
        return False

    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        logger.info("Testing health endpoint...")

        try:
            async with self.session.get(f"{self.base_url}/healthz") as response:
                health_data = await response.json()

                result = {
                    "endpoint": "health",
                    "status": "success",
                    "response_code": response.status,
                    "data": health_data,
                    "model_status": health_data.get("model_status", {}),
                }

                logger.info(f"Health check: {health_data.get('status')}")
                if health_data.get("model_status"):
                    model_info = health_data["model_status"]
                    logger.info(f"Models loaded: {model_info.get('models_loaded')}")
                    logger.info(f"Device: {model_info.get('device')}")
                    logger.info(f"Warmup complete: {model_info.get('warmup_complete')}")

                return result

        except Exception as e:
            logger.error(f"Health endpoint test failed: {e}")
            return {"endpoint": "health", "status": "failed", "error": str(e)}

    async def test_analyze_endpoint(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test analyze endpoint with a single text."""
        start_time = time.time()

        try:
            payload = {
                "text": text_data["text"],
                "context": None,
                "thread_id": f"test_{int(time.time())}",
            }

            async with self.session.post(
                f"{self.base_url}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:

                response_time = time.time() - start_time

                if response.status == 200:
                    result_data = await response.json()

                    # Validate response structure
                    required_fields = [
                        "toxicity",
                        "bullying",
                        "role",
                        "emotion",
                        "emotion_strength",
                        "scores",
                        "explanations",
                    ]

                    missing_fields = [
                        field for field in required_fields if field not in result_data
                    ]

                    result = {
                        "text_description": text_data["description"],
                        "input_text": text_data["text"],
                        "status": "success",
                        "response_code": response.status,
                        "response_time_s": round(response_time, 3),
                        "api_processing_time_ms": result_data.get("processing_time_ms", 0),
                        "predictions": {
                            "toxicity": result_data.get("toxicity"),
                            "bullying": result_data.get("bullying"),
                            "role": result_data.get("role"),
                            "emotion": result_data.get("emotion"),
                            "emotion_strength": result_data.get("emotion_strength"),
                        },
                        "scores": result_data.get("scores", {}),
                        "explanations": result_data.get("explanations", {}),
                        "validation": {
                            "missing_fields": missing_fields,
                            "valid_structure": len(missing_fields) == 0,
                        },
                        "expectations": {
                            "toxicity_match": (
                                result_data.get("toxicity") == text_data.get("expected_toxicity")
                            ),
                            "emotion_match": (
                                result_data.get("emotion") == text_data.get("expected_emotion")
                            ),
                            "bullying_match": (
                                result_data.get("bullying")
                                == text_data.get("expected_bullying", "none")
                            ),
                        },
                    }

                    # Log key predictions
                    logger.info(
                        f"Text: '{text_data['text'][:30]}...' -> "
                        f"Toxicity: {result_data.get('toxicity')} "
                        f"(expected: {text_data.get('expected_toxicity')}), "
                        f"Emotion: {result_data.get('emotion')} "
                        f"(expected: {text_data.get('expected_emotion')})"
                    )

                else:
                    error_data = await response.text()
                    result = {
                        "text_description": text_data["description"],
                        "input_text": text_data["text"],
                        "status": "failed",
                        "response_code": response.status,
                        "response_time_s": round(response_time, 3),
                        "error": error_data,
                    }
                    logger.error(f"API error {response.status}: {error_data}")

                return result

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Request failed for '{text_data['text']}': {e}")
            return {
                "text_description": text_data["description"],
                "input_text": text_data["text"],
                "status": "failed",
                "response_time_s": round(response_time, 3),
                "error": str(e),
            }

    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test metrics endpoint."""
        logger.info("Testing metrics endpoint...")

        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    metrics_data = await response.json()

                    result = {
                        "endpoint": "metrics",
                        "status": "success",
                        "data": metrics_data,
                    }

                    logger.info(
                        f"Total predictions: \
                        {metrics_data.get('total_predictions')}"
                    )
                    logger.info(
                        f"Success rate: \
                        {metrics_data.get('success_rate')}"
                    )
                    avg_time = metrics_data.get("average_processing_time_ms")
                    logger.info(f"Avg processing time: {avg_time}ms")

                    return result
                else:
                    error_data = await response.text()
                    return {
                        "endpoint": "metrics",
                        "status": "failed",
                        "error": error_data,
                    }

        except Exception as e:
            logger.error(f"Metrics endpoint test failed: {e}")
            return {"endpoint": "metrics", "status": "failed", "error": str(e)}

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive API test suite."""
        logger.info("Starting comprehensive API test suite...")

        # Wait for API to be ready
        if not await self.wait_for_api_startup():
            return {"status": "failed", "error": "API failed to start"}

        # Test health endpoint
        health_result = await self.test_health_endpoint()

        # Test analyze endpoint with all test cases
        analyze_results = []
        successful_predictions = 0
        failed_predictions = 0
        total_response_time = 0

        logger.info(
            f"Testing analyze endpoint with {len(TEST_TEXTS)} \
            examples..."
        )

        for i, text_data in enumerate(TEST_TEXTS):
            logger.info(
                f"Testing example {i+1}/{len(TEST_TEXTS)}: \
                {text_data['description']}"
            )
            result = await self.test_analyze_endpoint(text_data)
            analyze_results.append(result)

            if result["status"] == "success":
                successful_predictions += 1
                total_response_time += result["response_time_s"]
            else:
                failed_predictions += 1

            # Small delay between requests
            await asyncio.sleep(0.1)

        # Test metrics endpoint
        metrics_result = await self.test_metrics_endpoint()

        # Calculate overall statistics
        total_tests = len(TEST_TEXTS)
        success_rate = successful_predictions / total_tests if total_tests > 0 else 0
        avg_response_time = (
            total_response_time / successful_predictions if successful_predictions > 0 else 0
        )

        # Validate predictions accuracy
        prediction_accuracy = self._calculate_prediction_accuracy(analyze_results)

        test_summary = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_test_cases": total_tests,
                "api_base_url": self.base_url,
            },
            "overall_results": {
                "successful_requests": successful_predictions,
                "failed_requests": failed_predictions,
                "success_rate": round(success_rate, 4),
                "average_response_time_s": round(avg_response_time, 3),
            },
            "prediction_accuracy": prediction_accuracy,
            "health_check": health_result,
            "metrics": metrics_result,
            "detailed_results": analyze_results,
            "conclusion": self._generate_test_conclusion(
                success_rate, prediction_accuracy, health_result
            ),
        }

        return test_summary

    def _calculate_prediction_accuracy(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate prediction accuracy statistics."""
        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            return {"status": "no_successful_predictions"}

        toxicity_correct = sum(1 for r in successful_results if r["expectations"]["toxicity_match"])
        emotion_correct = sum(1 for r in successful_results if r["expectations"]["emotion_match"])
        bullying_correct = sum(1 for r in successful_results if r["expectations"]["bullying_match"])

        total = len(successful_results)

        return {
            "total_successful_predictions": total,
            "toxicity_accuracy": round(toxicity_correct / total, 4),
            "emotion_accuracy": round(emotion_correct / total, 4),
            "bullying_accuracy": round(bullying_correct / total, 4),
            "overall_accuracy": round(
                (toxicity_correct + emotion_correct + bullying_correct) / (total * 3), 4
            ),
        }

    def _generate_test_conclusion(
        self, success_rate: float, accuracy_data: Dict, health_data: Dict
    ) -> Dict[str, Any]:
        """Generate test conclusion and recommendations."""

        model_status = health_data.get("data", {}).get("model_status", {})
        models_loaded = model_status.get("models_loaded", False)

        if not models_loaded:
            return {
                "status": "critical_failure",
                "message": "Models not loaded - API not functional",
                "recommendations": [
                    "Check model files",
                ],
            }

        if success_rate < 0.5:
            return {
                "status": "major_issues",
                "message": "Success rate is below 50%, indicating major API issues",
                "recommendations": [
                    "Check API error logs",
                ],
            }

        if success_rate >= 0.9 and accuracy_data.get("overall_accuracy", 0) >= 0.6:
            return {
                "status": "excellent",
                "message": "API performance is excellent",
                "recommendations": [
                    "Monitor performance",
                ],
            }

        elif success_rate >= 0.8:
            return {
                "status": "good",
                "message": "API mostly functional with minor issues",
                "recommendations": [
                    "Investigate failed cases",
                ],
            }

        else:
            return {
                "status": "needs_improvement",
                "message": "API functional but needs optimization",
                "recommendations": [
                    "Review model accuracy",
                ],
            }


async def main():
    """Run the API integration test."""
    print("=" * 60)
    print("CyberPuppy API Integration Test")
    print("=" * 60)

    async with APITester() as tester:
        test_results = await tester.run_comprehensive_test()

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"api_test_results_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        overall = test_results["overall_results"]
        accuracy = test_results["prediction_accuracy"]
        conclusion = test_results["conclusion"]

        print(
            f"Total test cases: \
            {test_results['test_metadata']['total_test_cases']}"
        )
        print(f"Successful requests: {overall['successful_requests']}")
        print(f"Failed requests: {overall['failed_requests']}")
        print(f"Success rate: {overall['success_rate']:.1%}")
        print(
            f"Average response time: \
            {overall['average_response_time_s']:.3f}s"
        )

        if accuracy.get("total_successful_predictions", 0) > 0:
            print("\nPrediction Accuracy:")
            print(f"  Toxicity: {accuracy['toxicity_accuracy']:.1%}")
            print(f"  Emotion: {accuracy['emotion_accuracy']:.1%}")
            print(f"  Bullying: {accuracy['bullying_accuracy']:.1%}")
            print(f"  Overall: {accuracy['overall_accuracy']:.1%}")

        print(f"\nConclusion: {conclusion['status'].upper()}")
        print(f"Message: {conclusion['message']}")

        if conclusion.get("recommendations"):
            print("\nRecommendations:")
            for rec in conclusion["recommendations"]:
                print(f"  • {rec}")

        print(f"\nDetailed results saved to: {results_file}")
        print("=" * 60)

        return test_results


if __name__ == "__main__":
    asyncio.run(main())
