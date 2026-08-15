"""Run the whole offline test suite with the mock model: python -m tests.run_all"""

from tests import test_agent, test_tree

if __name__ == "__main__":
    test_agent._run()
    print()
    test_tree._run()
    print("\n=== ALL SUITES PASSED ===")
