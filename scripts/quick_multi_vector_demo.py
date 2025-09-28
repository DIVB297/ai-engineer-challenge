#!/usr/bin/env python3
"""
Quick Multi-Vector Demo

This script demonstrates the difference between cosine similarity and dot product similarity
in a simple, easy-to-understand way.

Usage:
    python scripts/quick_multi_vector_demo.py
"""

import time

import requests


def test_similarity_metrics(base_url="http://localhost:5000"):
    """Test both similarity metrics with the same query"""

    # Test query
    query = "What is machine learning?"
    user_id = "demo_user"

    print("🔬 Multi-Vector Similarity Demo")
    print("=" * 50)
    print(f"Query: {query}")
    print()

    # Test cosine similarity
    print("📊 Testing Cosine Similarity...")
    cosine_start = time.time()

    try:
        cosine_response = requests.post(
            f"{base_url}/chat",
            json={
                "user_id": user_id,
                "query": query,
                "k": 3,
                "similarity_metric": "cosine",
            },
        )

        cosine_time = time.time() - cosine_start

        if cosine_response.status_code == 200:
            cosine_data = cosine_response.json()
            print(f"✅ Success! ({cosine_time:.2f}s)")
            print(f"   Answer length: {len(cosine_data.get('answer', ''))} chars")
            print(f"   Sources found: {len(cosine_data.get('source_docs', []))}")

            # Show top scores
            for i, doc in enumerate(cosine_data.get("source_docs", [])[:3]):
                print(f"   Score #{i + 1}: {doc.get('score', 0):.4f}")
        else:
            print(f"❌ Failed with status: {cosine_response.status_code}")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print()

    # Test dot product similarity
    print("⚡ Testing Dot Product Similarity...")
    dot_start = time.time()

    try:
        dot_response = requests.post(
            f"{base_url}/chat",
            json={
                "user_id": user_id,
                "query": query,
                "k": 3,
                "similarity_metric": "dot_product",
            },
        )

        dot_time = time.time() - dot_start

        if dot_response.status_code == 200:
            dot_data = dot_response.json()
            print(f"✅ Success! ({dot_time:.2f}s)")
            print(f"   Answer length: {len(dot_data.get('answer', ''))} chars")
            print(f"   Sources found: {len(dot_data.get('source_docs', []))}")

            # Show top scores
            for i, doc in enumerate(dot_data.get("source_docs", [])[:3]):
                print(f"   Score #{i + 1}: {doc.get('score', 0):.4f}")
        else:
            print(f"❌ Failed with status: {dot_response.status_code}")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print()
    print("📈 Comparison Summary:")
    print(f"   Cosine similarity timing: {cosine_time:.2f}s")
    print(f"   Dot product timing: {dot_time:.2f}s")
    print(f"   Performance difference: {abs(cosine_time - dot_time):.2f}s")

    # Compare scores
    cosine_scores = [doc.get("score", 0) for doc in cosine_data.get("source_docs", [])]
    dot_scores = [doc.get("score", 0) for doc in dot_data.get("source_docs", [])]

    if cosine_scores and dot_scores:
        avg_cosine = sum(cosine_scores) / len(cosine_scores)
        avg_dot = sum(dot_scores) / len(dot_scores)
        print(f"   Average cosine score: {avg_cosine:.4f}")
        print(f"   Average dot product score: {avg_dot:.4f}")
        print(f"   Score difference: {abs(avg_cosine - avg_dot):.4f}")

    print()
    print("🎯 Key Differences:")
    print("   • Cosine similarity: Measures angle between vectors (0-1 range)")
    print("   • Dot product: Measures magnitude and direction (unbounded)")
    print("   • Cosine is normalized, dot product considers vector magnitudes")
    print("   • Use cosine for semantic similarity, dot product for importance weighting")


def test_direct_embedding_service(base_url="http://localhost:8000"):
    """Test the embedding service directly"""
    print("\n🔧 Testing Direct Embedding Service...")
    print("=" * 50)

    query = "artificial intelligence"

    for metric in ["cosine", "dot_product"]:
        print(f"\n📡 Testing {metric} via embedding service...")

        try:
            response = requests.get(
                f"{base_url}/search",
                params={"query": query, "k": 3, "similarity_metric": metric},
            )

            if response.status_code == 200:
                data = response.json()
                print("✅ Success!")
                print(f"   Processing time: {data.get('processing_time_ms', 0):.2f}ms")
                print(f"   Results: {len(data.get('results', []))}")

                for i, result in enumerate(data.get("results", [])[:2]):
                    print(f"   Result #{i + 1}: score={result.get('score', 0):.4f}")
            else:
                print(f"❌ Failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run the demo"""
    print("🚀 Starting Multi-Vector Similarity Demo")
    print("This demo requires the RAG system to be running on localhost")
    print()

    # Check if services are running
    try:
        health_response = requests.get("http://localhost:5000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Orchestrator service not responding. Please run:")
            print("   docker-compose up -d")
            return
    except Exception:
        print("❌ Services not running. Please start them first:")
        print("   docker-compose up -d")
        return

    # Run the demo
    test_similarity_metrics()
    test_direct_embedding_service()

    print("\n🎉 Demo completed!")
    print("To explore more:")
    print("   • Try the React demo UI at http://localhost:5000")
    print("   • Run the comprehensive test: python scripts/test_multi_vector.py")
    print("   • Check Prometheus metrics at http://localhost:8000/metrics")


if __name__ == "__main__":
    main()
