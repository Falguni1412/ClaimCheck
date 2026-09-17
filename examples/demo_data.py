"""
Demo test cases for ClaimCheck
"""
import requests

API_URL = "http://localhost:8000"


def test_medical_hallucination():
    """
    Test case: LLM answer about metformin with a hallucinated statistic
    Expected: "Lactic acidosis occurs in 10% of patients" should be CONTRADICTED
    """
    print("=" * 70)
    print("TEST 1: Medical Hallucination Detection")
    print("=" * 70)

    payload = {
        "answer": "Metformin should be taken on an empty stomach. It can be combined with insulin therapy. Lactic acidosis occurs in 10% of patients. Nausea is a common side effect.",
        "sources": [
            "Metformin is a medication for type 2 diabetes. Take metformin with meals to reduce stomach upset. Do not take on an empty stomach.",
            "Common side effects of metformin include nausea, diarrhea, and stomach pain. Lactic acidosis is a rare but serious side effect, occurring in approximately 1 in 30,000 patient-years.",
            "Metformin can be safely combined with insulin therapy under medical supervision. This combination is commonly prescribed for patients with poorly controlled diabetes."
        ]
    }

    response = requests.post(f"{API_URL}/verify", json=payload)
    data = response.json()

    print(f"\nRisk Score: {data['risk_score'] * 100:.0f}%")
    print(f"Summary: {data['summary']}")
    print(f"\nClaims analyzed:")
    for i, claim in enumerate(data['claims'], 1):
        emoji = "✅" if claim['verdict'] == "SUPPORTED" else "❌" if claim['verdict'] == "CONTRADICTED" else "⚠️"
        print(f"  {i}. {emoji} {claim['verdict']} ({claim['confidence']:.0%}): {claim['claim']}")

    return data


def test_company_policy():
    """
    Test case: LLM answer about refund policy
    Expected: "Processed within 24 hours" should be CONTRADICTED
    """
    print("\n" + "=" * 70)
    print("TEST 2: Company Policy Verification")
    print("=" * 70)

    payload = {
        "answer": "Refunds are processed within 24 hours. You will receive a confirmation email within 30 minutes. Your account will be automatically credited. You can track the refund status online.",
        "sources": [
            "Our refund policy states that refunds are processed within 5-7 business days. A confirmation email is sent immediately after the refund is initiated. The refunded amount will be credited to your original payment method within 10 business days.",
            "You can track your refund status by logging into your account and visiting the Orders section. Refund history is available for up to 12 months."
        ]
    }

    response = requests.post(f"{API_URL}/verify", json=payload)
    data = response.json()

    print(f"\nRisk Score: {data['risk_score'] * 100:.0f}%")
    print(f"Summary: {data['summary']}")
    print(f"\nClaims analyzed:")
    for i, claim in enumerate(data['claims'], 1):
        emoji = "✅" if claim['verdict'] == "SUPPORTED" else "❌" if claim['verdict'] == "CONTRADICTED" else "⚠️"
        print(f"  {i}. {emoji} {claim['verdict']} ({claim['confidence']:.0%}): {claim['claim']}")

    return data


def test_coding_hallucination():
    """
    Test case: LLM suggesting a non-existent Python function
    Expected: Claim about a function should be UNVERIFIABLE (or CONTRADICTED if doc says it doesn't exist)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Code Generation Verification")
    print("=" * 70)

    payload = {
        "answer": "Python's requests library has a function called fetch_url() that works like requests.get(). It can also use requests.post() for sending data. The library supports JSON parsing with requests.json().",
        "sources": [
            "The requests library is a popular HTTP client for Python. Main functions include requests.get() for GET requests, requests.post() for POST requests, and requests.put() for PUT requests. The library does not have a fetch_url() function. JSON responses are accessed via response.json() method, not requests.json()."
        ]
    }

    response = requests.post(f"{API_URL}/verify", json=payload)
    data = response.json()

    print(f"\nRisk Score: {data['risk_score'] * 100:.0f}%")
    print(f"Summary: {data['summary']}")
    print(f"\nClaims analyzed:")
    for i, claim in enumerate(data['claims'], 1):
        emoji = "✅" if claim['verdict'] == "SUPPORTED" else "❌" if claim['verdict'] == "CONTRADICTED" else "⚠️"
        print(f"  {i}. {emoji} {claim['verdict']} ({claim['confidence']:.0%}): {claim['claim']}")

    return data


if __name__ == "__main__":
    print("\n🔍 ClaimCheck Demo Test Cases\n")

    try:
        # Check if API is running
        health = requests.get(f"{API_URL}/")
        print(f"API Status: {health.json()['status']}\n")

        # Run all tests
        test_medical_hallucination()
        test_company_policy()
        test_coding_hallucination()

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to ClaimCheck API")
        print("Make sure the backend is running: uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
