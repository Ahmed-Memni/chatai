# test_response.py
from src.chains import full_chain

def test_full_chain():
    # Mock question for testing
    mock_question = "List all client emails"

    # Run the full chain
    result = full_chain.run(mock_question)

    # Print the result
    print("=== FullChain Output ===")
    print(result["output"])

if __name__ == "__main__":
    test_full_chain()