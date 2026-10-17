"""Test runner for all paratransit routing unit tests."""
from __future__ import annotations

import unittest
import os
import sys

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
test_dir = os.path.join(parent_dir, "tests")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def run_suite():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"PARATRANSIT TEST RESULTS:")
    print(f"  Tests Run:     {result.testsRun}")
    print(f"  Errors:        {len(result.errors)}")
    print(f"  Failures:      {len(result.failures)}")
    print(f"  Success:       {result.wasSuccessful()}")
    print("=" * 60 + "\n")

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
