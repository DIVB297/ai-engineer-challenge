#!/usr/bin/env python3
"""
Multi-Vector Similarity Testing Script

This script demonstrates the multi-vector similarity support in the RAG system,
comparing cosine similarity vs dot product similarity for the same queries.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiVectorTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.embedding_url = "http://localhost:8000"

    async def test_similarity_metrics(self, queries: List[str]) -> Dict[str, Any]:
        """Test both cosine and dot product similarity for given queries"""
        results = {"test_queries": queries, "comparisons": [], "summary": {}}

        async with aiohttp.ClientSession() as session:
            for query in queries:
                logger.info(f"Testing query: {query}")

                # Test cosine similarity
                cosine_result = await self._make_chat_request(
                    session, query, similarity_metric="cosine"
                )

                # Test dot product similarity
                dot_product_result = await self._make_chat_request(
                    session, query, similarity_metric="dot_product"
                )

                comparison = {
                    "query": query,
                    "cosine": {
                        "timing_ms": cosine_result.get("timing_ms", 0),
                        "sources": len(cosine_result.get("source_docs", [])),
                        "top_scores": [
                            doc.get("score", 0) for doc in cosine_result.get("source_docs", [])[:3]
                        ],
                        "answer_length": len(cosine_result.get("answer", "")),
                    },
                    "dot_product": {
                        "timing_ms": dot_product_result.get("timing_ms", 0),
                        "sources": len(dot_product_result.get("source_docs", [])),
                        "top_scores": [
                            doc.get("score", 0)
                            for doc in dot_product_result.get("source_docs", [])[:3]
                        ],
                        "answer_length": len(dot_product_result.get("answer", "")),
                    },
                    "score_difference": self._calculate_score_difference(
                        cosine_result.get("source_docs", []),
                        dot_product_result.get("source_docs", []),
                    ),
                }

                results["comparisons"].append(comparison)

                # Add a small delay between requests
                await asyncio.sleep(0.5)

        # Generate summary
        results["summary"] = self._generate_summary(results["comparisons"])
        return results

    async def _make_chat_request(
        self, session: aiohttp.ClientSession, query: str, similarity_metric: str
    ) -> Dict[str, Any]:
        """Make a chat request with specified similarity metric"""
        try:
            async with session.post(
                f"{self.base_url}/chat",
                json={
                    "user_id": "multi_vector_test",
                    "query": query,
                    "k": 5,
                    "similarity_metric": similarity_metric,
                },
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Request failed with status {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return {}

    def _calculate_score_difference(
        self, cosine_docs: List[Dict], dot_product_docs: List[Dict]
    ) -> Dict[str, float]:
        """Calculate differences between similarity scores"""
        if not cosine_docs or not dot_product_docs:
            return {"avg_difference": 0.0, "max_difference": 0.0}

        differences = []
        for i, (cos_doc, dot_doc) in enumerate(zip(cosine_docs, dot_product_docs)):
            cos_score = cos_doc.get("score", 0)
            dot_score = dot_doc.get("score", 0)
            differences.append(abs(cos_score - dot_score))

        return {
            "avg_difference": (sum(differences) / len(differences) if differences else 0.0),
            "max_difference": max(differences) if differences else 0.0,
        }

    def _generate_summary(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not comparisons:
            return {}

        cosine_times = [c["cosine"]["timing_ms"] for c in comparisons]
        dot_product_times = [c["dot_product"]["timing_ms"] for c in comparisons]

        return {
            "total_queries_tested": len(comparisons),
            "average_timing": {
                "cosine_ms": (sum(cosine_times) / len(cosine_times) if cosine_times else 0),
                "dot_product_ms": (
                    sum(dot_product_times) / len(dot_product_times) if dot_product_times else 0
                ),
            },
            "score_analysis": {
                "avg_score_differences": [
                    c["score_difference"]["avg_difference"] for c in comparisons
                ],
                "max_score_differences": [
                    c["score_difference"]["max_difference"] for c in comparisons
                ],
            },
        }

    async def test_direct_embedding_search(self, query: str) -> Dict[str, Any]:
        """Test direct embedding service search with both metrics"""
        results = {}

        async with aiohttp.ClientSession() as session:
            for metric in ["cosine", "dot_product"]:
                try:
                    async with session.get(
                        f"{self.embedding_url}/search",
                        params={"query": query, "k": 5, "similarity_metric": metric},
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            results[metric] = {
                                "processing_time_ms": data.get("processing_time_ms", 0),
                                "results_count": len(data.get("results", [])),
                                "top_scores": [
                                    r.get("score", 0) for r in data.get("results", [])[:3]
                                ],
                            }
                        else:
                            logger.error(f"Direct search failed for {metric}: {response.status}")
                            results[metric] = {"error": f"HTTP {response.status}"}
                except Exception as e:
                    logger.error(f"Error in direct search for {metric}: {e}")
                    results[metric] = {"error": str(e)}

        return results


async def main():
    """Run the multi-vector similarity tests"""
    tester = MultiVectorTester()

    # Test queries that should show different similarity behaviors
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain vector databases",
        "What is the difference between AI and ML?",
        "How does gradient descent optimization work?",
    ]

    logger.info("Starting multi-vector similarity tests...")

    # Test full RAG pipeline
    logger.info("Testing full RAG pipeline with both similarity metrics...")
    rag_results = await tester.test_similarity_metrics(test_queries)

    # Test direct embedding service
    logger.info("Testing direct embedding service search...")
    direct_results = await tester.test_direct_embedding_search(test_queries[0])

    # Save results
    timestamp = int(time.time())
    results_file = f"multi_vector_test_results_{timestamp}.json"

    final_results = {
        "timestamp": timestamp,
        "test_type": "multi_vector_similarity_comparison",
        "rag_pipeline_results": rag_results,
        "direct_embedding_results": direct_results,
        "configuration": {
            "orchestrator_url": tester.base_url,
            "embedding_service_url": tester.embedding_url,
            "test_queries_count": len(test_queries),
        },
    }

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Test results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-VECTOR SIMILARITY TEST RESULTS")
    print("=" * 80)

    if rag_results.get("summary"):
        summary = rag_results["summary"]
        print(f"Total queries tested: {summary.get('total_queries_tested', 0)}")

        avg_timing = summary.get("average_timing", {})
        print(f"Average timing - Cosine: {avg_timing.get('cosine_ms', 0):.2f}ms")
        print(f"Average timing - Dot Product: {avg_timing.get('dot_product_ms', 0):.2f}ms")

        score_analysis = summary.get("score_analysis", {})
        avg_diffs = score_analysis.get("avg_score_differences", [])
        if avg_diffs:
            print(f"Average score difference: {sum(avg_diffs) / len(avg_diffs):.4f}")

    print(f"\nDetailed results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
