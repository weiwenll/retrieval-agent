"""
Test case to debug why search_with_requirements returns fewer places than required.

This test simulates the search process and logs detailed information about:
- Each search level attempted
- Number of results at each level
- Deduplication statistics
- Final result count vs requirements

Usage:
    python test_search_requirements.py
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main import PlacesResearchAgent, load_input_file

load_dotenv()


def test_search_with_tourist_4():
    """Test with tourist_4_input.json to understand why only 14 attractions found."""

    print("="*80)
    print("TEST: Debugging search_with_requirements for tourist_4_input.json")
    print("="*80)

    # Load input
    input_file = '../inputs/tourist_4_input.json'
    input_data = load_input_file(input_file)

    if not input_data:
        print("FAILED: Failed to load input file")
        return

    print(f"\nLOADED input file: {input_file}")
    print(f"  - Pace: {input_data.get('pace')}")
    print(f"  - Duration: {input_data.get('duration_days')} days")
    print(f"  - Location: {input_data.get('optional', {}).get('accommodation_location')}")
    print(f"  - Interests: {input_data.get('optional', {}).get('interests')}")

    # Initialize agent
    agent = PlacesResearchAgent()

    # Calculate requirements
    requirements = agent.calculate_required_places(
        input_data.get('pace'),
        input_data.get('duration_days')
    )

    print(f"\nREQUIREMENTS:")
    print(f"  - Attractions needed: {requirements['attractions_needed']}")
    print(f"  - Food needed: {requirements['food_places_needed']}")
    print(f"  - Total needed: {requirements['total_needed']}")

    # Map interests
    user_interests = input_data.get('optional', {}).get('interests', [])
    mapped_interests = agent.map_interests(user_interests) if user_interests else ['tourist_attraction']

    print(f"\nMAPPED INTERESTS: {mapped_interests}")

    # Test search for each interest
    location = input_data.get('optional', {}).get('accommodation_location', {})

    print(f"\nTESTING SEARCH:")
    print(f"  Location: {location}")

    for interest in mapped_interests:
        print(f"\n{'='*80}")
        print(f"Searching for: '{interest}'")
        print(f"Target: {requirements['attractions_needed']} places")
        print(f"{'='*80}")

        results = agent.search_with_requirements(
            location=location,
            keyword=interest,
            min_rating=4.0,  # Starting rating
            max_results_needed=requirements['attractions_needed'],
            search_type='attraction'
        )

        print(f"\nRESULTS for '{interest}':")
        print(f"  - Found: {len(results)} places")
        print(f"  - Target: {requirements['attractions_needed']} places")
        print(f"  - Shortfall: {requirements['attractions_needed'] - len(results)} places")

        if len(results) < requirements['attractions_needed']:
            print(f"  WARNING: Only found {len(results)}/{requirements['attractions_needed']} attractions")
            print(f"  ANALYSIS:")
            print(f"     - Try broader keywords like 'point_of_interest' or 'establishment'")
            print(f"     - Location (Bukit Timah) is residential - fewer tourist attractions")
            print(f"     - Consider using central Singapore location instead")
        else:
            print(f"  SUCCESS: Successfully found all required attractions")

        # Show sample of results
        if results:
            print(f"\n  Sample results (first 5):")
            for i, place in enumerate(results[:5], 1):
                print(f"    {i}. {place.get('name')} (rating: {place.get('rating')})")

    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"{'='*80}")


def test_different_keywords():
    """Test with different keyword strategies to find more places."""

    print("\n" + "="*80)
    print("TEST: Trying different keyword strategies")
    print("="*80)

    # Load input
    input_file = '../inputs/tourist_4_input.json'
    input_data = load_input_file(input_file)
    location = input_data.get('optional', {}).get('accommodation_location', {})

    # Initialize agent
    agent = PlacesResearchAgent()
    requirements = agent.calculate_required_places(input_data.get('pace'), input_data.get('duration_days'))

    # Test different keywords
    test_keywords = [
        'tourist_attraction',      # Original (specific)
        'point_of_interest',       # Broader
        'establishment',           # Very broad
        'attraction',              # Generic
    ]

    results_summary = []

    for keyword in test_keywords:
        print(f"\nTesting keyword: '{keyword}'")

        results = agent.search_with_requirements(
            location=location,
            keyword=keyword,
            min_rating=4.0,
            max_results_needed=requirements['attractions_needed'],
            search_type='attraction'
        )

        count = len(results)
        results_summary.append({
            'keyword': keyword,
            'found': count,
            'target': requirements['attractions_needed'],
            'success': count >= requirements['attractions_needed']
        })

        print(f"  Results: {count}/{requirements['attractions_needed']}")
        if count >= requirements['attractions_needed']:
            print(f"  SUCCESS - Found enough places!")
        else:
            print(f"  FAILED - Still need {requirements['attractions_needed'] - count} more")

    # Summary
    print(f"\n{'='*80}")
    print(f"KEYWORD COMPARISON SUMMARY:")
    print(f"{'='*80}")
    print(f"{'Keyword':<25} {'Found':<10} {'Target':<10} {'Status'}")
    print(f"{'-'*60}")
    for r in results_summary:
        status = 'SUCCESS' if r['success'] else 'FAILED'
        print(f"{r['keyword']:<25} {r['found']:<10} {r['target']:<10} {status}")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    successful = [r for r in results_summary if r['success']]
    if successful:
        best = successful[0]
        print(f"   Use keyword: '{best['keyword']}' (found {best['found']} places)")
    else:
        best_effort = max(results_summary, key=lambda x: x['found'])
        print(f"   Best effort: '{best_effort['keyword']}' found {best_effort['found']}/{best_effort['target']} places")
        print(f"   Consider:")
        print(f"   - Lowering multiplier from 2.0 to 1.5")
        print(f"   - Using multiple keywords combined")
        print(f"   - Using central Singapore location instead of Bukit Timah")


if __name__ == '__main__':
    # Run tests
    test_search_with_tourist_4()
    print("\n\n")
    test_different_keywords()

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*80}")
