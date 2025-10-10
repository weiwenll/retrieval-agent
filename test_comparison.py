"""
Compare direct tool call vs research_places function for tourist_4_input.json
"""
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append('ResearchAgent')

from ResearchAgent.tools import search_places

print("=" * 80)
print("TEST 1: Direct tool call with 'tourist attractions' keyword")
print("=" * 80)

start_time = time.time()
keyword = "tourist attractions"
results = search_places(
    location={"lat": 1.3294, "lon": 103.8021},
    radius=5000,
    keyword=keyword,
    min_rating=4.0
)
elapsed = time.time() - start_time

print(f"\nKeyword: '{keyword}'")
print(f"Location: lat=1.3294, lon=103.8021 (Bukit Timah)")
print(f"Radius: 5000m")
print(f"Min rating: 4.0")
print(f"Results found: {len(results)}")
print(f"Time elapsed: {elapsed:.2f} seconds")

if results:
    print("\nFirst 5 results:")
    for i, place in enumerate(results[:5], 1):
        print(f"  {i}. {place.get('name')} - Rating: {place.get('rating')}")

print("\n" + "=" * 80)
print("TEST 2: Via map_interests() with 'tourist attractions' input")
print("=" * 80)

# Test the mapping
from ResearchAgent.main import PlacesResearchAgent

agent = PlacesResearchAgent()
user_input = ["tourist attractions"]
mapped = agent.map_interests(user_input)

print(f"\nUser input: {user_input}")
print(f"Mapped interests: {mapped}")

# Now search with mapped interest
print(f"\nSearching with mapped keyword: '{mapped[0]}'")
start_time = time.time()
results2 = search_places(
    location={"lat": 1.3294, "lon": 103.8021},
    radius=5000,
    keyword=mapped[0],
    min_rating=4.0
)
elapsed2 = time.time() - start_time

print(f"Results found: {len(results2)}")
print(f"Time elapsed: {elapsed2:.2f} seconds")

if results2:
    print("\nFirst 5 results:")
    for i, place in enumerate(results2[:5], 1):
        print(f"  {i}. {place.get('name')} - Rating: {place.get('rating')}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Direct call with 'tourist attractions': {len(results)} results")
print(f"Via map_interests with 'tourist attractions': {len(results2)} results")
print(f"Keywords used: '{keyword}' vs '{mapped[0]}'")

if len(results) != len(results2):
    print(f"\n⚠ DIFFERENCE FOUND: {abs(len(results) - len(results2))} result difference")
else:
    print(f"\n✓ Same number of results")

print("\n" + "=" * 80)
print("TEST 3: Full research_places() flow")
print("=" * 80)

from ResearchAgent.main import research_places

start_time = time.time()
full_results = research_places('inputs/tourist_4_input.json')
elapsed3 = time.time() - start_time

print(f"\nFull research_places() execution:")
print(f"Attractions found: {full_results.get('attractions_count')}")
print(f"Food found: {full_results.get('food_count')}")
print(f"Total places: {full_results.get('places_found')}")
print(f"Requirements: {full_results.get('requirements')}")
print(f"Time elapsed: {elapsed3:.2f} seconds")
