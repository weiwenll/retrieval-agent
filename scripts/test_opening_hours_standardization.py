"""
Test script for opening hours standardization.

Tests the standardize_opening_hours function with various input formats.
"""

import sys
import os

# Add ResearchAgent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ResearchAgent'))

from tools import standardize_opening_hours


def test_standardization():
    """Test various opening hours formats"""

    test_cases = [
        # (input, expected_output, description)
        ("10:00-19:00", "10:00-19:00", "Already standardized"),
        ("07:00-12:00,18:00-21:00", "07:00-12:00,18:00-21:00", "Already standardized with multiple periods"),
        ("12:002:30PM, 3:0010:00PM", "12:00-14:30,15:00-22:00", "Missing dashes and AM/PM conversion"),
        ("10:00AM7:00PM", "10:00-19:00", "Missing dash with AM/PM"),
        ("11:30AM4:00PM, 6:0011:30PM", "11:30-16:00,18:00-23:30", "Multiple periods with mixed formats"),
        ("6:3010:30PM", "06:30-22:30", "Missing dash and AM/PM (single period)"),
        ("All Day", "00:00-23:59", "All day text"),
        ("Open 24 hours", "00:00-23:59", "Open 24 hours text"),
        ("Closed", "Closed", "Closed"),
        ("6:30PM12:00AM", "18:30-00:00", "PM to midnight conversion"),
        ("10:00 AM – 7:00 PM", "10:00-19:00", "Standard Google format with dash"),
        ("9:00 AM – 5:00 PM", "09:00-17:00", "Standard format with single digit hour"),
        (None, "00:00-23:59", "None defaults to open all day"),
        ("", "00:00-23:59", "Empty string defaults to open all day"),
    ]

    print("Testing opening hours standardization\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for input_str, expected, description in test_cases:
        result = standardize_opening_hours(input_str)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] - {description}")
        print(f"  Input:    '{input_str}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")

    print("\n" + "=" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    return failed == 0


if __name__ == "__main__":
    success = test_standardization()
    sys.exit(0 if success else 1)
