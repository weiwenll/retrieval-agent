import os
import time
import unicodedata
from typing import Optional, Dict, List, Set
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Import mappings from config
from config import UNINTEREST_MAPPINGS, SPECIAL_INTEREST_CATEGORIES, DIETARY_EXCLUSIONS


def remove_unicode(text: str) -> str:
    """
    Remove or replace unicode characters with ASCII equivalents.

    Args:
        text: String that may contain unicode characters

    Returns:
        ASCII-safe string
    """
    if not text:
        return text

    # Normalize unicode to closest ASCII representation
    # NFD = Canonical Decomposition
    normalized = unicodedata.normalize('NFD', text)

    # Remove non-ASCII characters
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')

    return ascii_text


def standardize_opening_hours(hours_str: str) -> str:
    """
    Standardize opening hours to 24-hour format with proper separators.

    Handles various formats:
    - "10:00AM7:00PM" → "10:00-19:00"
    - "12:002:30PM, 3:0010:00PM" → "12:00-14:30,15:00-22:00"
    - "6:30PM12:00AM" → "18:30-00:00"
    - "Open 24 hours" or "All Day" → "00:00-23:59"
    - "Closed" → "Closed"
    - None or empty string → "00:00-23:59" (default to open all day)

    Args:
        hours_str: Original hours string from Google API

    Returns:
        Standardized hours string in 24-hour format
    """
    import re

    # NOTE: If no opening hours provided (None or empty), default to open all day
    # This ensures places without explicit hours are still usable in the itinerary
    if not hours_str:
        return "00:00-23:59"

    # Handle special cases
    hours_lower = hours_str.lower().strip()
    if "open 24 hours" in hours_lower or "all day" in hours_lower or hours_lower == "24 hours":
        return "00:00-23:59"
    if "closed" in hours_lower:
        return "Closed"

    # Replace various dash characters with standard dash
    hours_str = re.sub(r'[\u2013\u2014\u2212\u002d]', '-', hours_str)  # En dash, em dash, minus, hyphen
    hours_str = hours_str.replace('–', '-').replace('—', '-')

    # Helper function to convert 12-hour time to 24-hour format
    def convert_to_24h(time_str: str, inferred_meridiem: str = None) -> str:
        """Convert time like '10:00AM' or '2:30PM' to '10:00' or '14:30'"""
        time_str = time_str.strip()

        # Extract time and AM/PM
        match = re.match(r'^(\d{1,2}):?(\d{2})\s*(AM|PM)?$', time_str, re.IGNORECASE)
        if not match:
            # Fix missing colon if needed (e.g., "1200" → "12:00")
            if ':' not in time_str and len(time_str) >= 3:
                if len(time_str) == 3:  # e.g., "300" → "3:00"
                    time_str = f"{time_str[0]}:{time_str[1:3]}"
                else:  # e.g., "1200" → "12:00"
                    time_str = f"{time_str[:-2]}:{time_str[-2:]}"
            # If still no AM/PM and we have inferred, use it
            if inferred_meridiem:
                time_str = time_str + inferred_meridiem
            match = re.match(r'^(\d{1,2}):?(\d{2})\s*(AM|PM)?$', time_str, re.IGNORECASE)
            if not match:
                return time_str

        hour_str, minute, meridiem = match.groups()
        hour = int(hour_str)
        minute = int(minute)

        # If no meridiem specified, use inferred or assume already 24-hour
        if not meridiem:
            meridiem = inferred_meridiem

        # Convert to 24-hour format
        if meridiem:
            meridiem = meridiem.upper()
            if meridiem == 'PM' and hour != 12:
                hour += 12
            elif meridiem == 'AM' and hour == 12:
                hour = 0

        return f"{hour:02d}:{minute:02d}"

    # Helper function to fix malformed time strings
    def fix_malformed_time(text: str) -> str:
        """Fix malformed times like '12:002:30PM' → '12:00-2:30PM'"""
        # Pattern: HH:MMHH:MMAM/PM (missing dash between times)
        text = re.sub(r'(\d{1,2}:\d{2})(\d{1,2}:\d{2})', r'\1-\2', text)

        # Pattern: HH:MMAM/PMHH:MMAM/PM (missing dash, with AM/PM)
        text = re.sub(r'(AM|PM)(\d{1,2}:\d{2})', r'\1-\2', text, flags=re.IGNORECASE)

        # Pattern: HHMMHH:MM (missing colon in first time)
        text = re.sub(r'(\d{1,2})(\d{2})-(\d{1,2}:\d{2})', r'\1:\2-\3', text)

        # Pattern: HH:MMHHMM (missing colon in second time)
        text = re.sub(r'(\d{1,2}:\d{2})-(\d{1,2})(\d{2})', r'\1-\2:\3', text)

        return text

    # Fix malformed separators first
    hours_str = fix_malformed_time(hours_str)

    # Split by comma to handle multiple time periods
    periods = [p.strip() for p in hours_str.split(',')]
    standardized_periods = []

    for period in periods:
        # Pattern for time range: "10:00AM-7:00PM" or "10:00AM7:00PM" or "10:00 AM - 7:00 PM"
        # Try to find two times with optional dash
        match = re.match(
            r'^(\d{1,2}:?\d{0,2})\s*(AM|PM)?\s*-\s*(\d{1,2}:?\d{0,2})\s*(AM|PM)?$',
            period,
            re.IGNORECASE
        )

        if match:
            start_time, start_meridiem, end_time, end_meridiem = match.groups()

            # Infer meridiem if missing
            # If end has PM and start doesn't, start is likely AM
            # If end has AM and start doesn't, both are probably AM
            # If start has meridiem but end doesn't, and end time < start time, end is likely PM
            inferred_start = start_meridiem
            inferred_end = end_meridiem

            if not start_meridiem and end_meridiem:
                # Infer start based on end
                end_hour = int(re.match(r'^(\d{1,2})', end_time).group(1))
                start_hour = int(re.match(r'^(\d{1,2})', start_time).group(1))

                if end_meridiem.upper() == 'PM':
                    # If end is PM, determine if start is AM or PM
                    # Heuristic: Use end time to determine context
                    # - End time 11 PM or later: likely dinner service (PM-PM), e.g., 6PM-11:30PM
                    # - End time 10:30 PM or earlier: likely all-day operation (AM-PM), e.g., 6:30AM-10:30PM

                    if start_hour > end_hour:
                        # Wrapping hours: 6PM-11:30PM (6>11 not true), or 11PM-2AM
                        # This case: start > end means like "11-2" which must be PM-AM wrap
                        # But for 6-11, this won't trigger. Let me reconsider.
                        # Actually start_hour=6, end_hour=11, so 6 is NOT > 11
                        inferred_start = 'PM'
                    elif start_hour < 6:
                        # Early morning hours: 3:00-10:00PM likely means 3PM-10PM
                        inferred_start = 'PM'
                    else:
                        # start_hour is 6-12
                        # Check the end_hour to determine context:
                        # If end is 11 or 12 (late night), likely dinner service PM-PM
                        # If end is 10 or earlier, likely all-day AM-PM
                        if end_hour >= 11:
                            # Late night end: 6:00-11:30PM = 6PM-11:30PM (dinner)
                            inferred_start = 'PM'
                        else:
                            # Earlier end: 6:30-10:30PM = 6:30AM-10:30PM (all-day)
                            inferred_start = 'AM'
                else:  # End is AM
                    inferred_start = 'AM'

            if start_meridiem and not end_meridiem:
                # Infer end based on start
                end_hour = int(re.match(r'^(\d{1,2})', end_time).group(1))
                start_hour = int(re.match(r'^(\d{1,2})', start_time).group(1))

                if start_meridiem.upper() == 'AM':
                    # If start is AM, end is likely PM if end_hour < start_hour
                    if end_hour < start_hour:
                        inferred_end = 'PM'
                    else:
                        inferred_end = 'AM'
                else:  # Start is PM
                    inferred_end = 'PM'

            # Convert both times to 24-hour format
            start_24h = convert_to_24h(start_time + (start_meridiem or ''), inferred_start)
            end_24h = convert_to_24h(end_time + (end_meridiem or ''), inferred_end)

            standardized_periods.append(f"{start_24h}-{end_24h}")
        else:
            # If pattern doesn't match, return original
            standardized_periods.append(period)

    # Join periods with comma
    return ','.join(standardized_periods)

def search_places(location, radius=2000, included_types=None, excluded_types=None, language='en', max_results=20, min_rating=0.0, destination_city=None):
    """
    Search Google Places nearby using the new Places API (v1).
    Returns RAW API data without formatting.

    Args:
        location: (lat, lng) tuple or dict with 'lat'/'lng' keys
        radius: search radius in meters (default 2000m, max 35000m)
        included_types: Single type (str) or multiple types (list) to search for
        excluded_types: List of place types to exclude (optional)
        language: language code (default 'en')
        max_results: maximum results to fetch (default 20)
        min_rating: minimum rating filter (0.0-5.0)
        destination_city: City name to filter results by (e.g., "Singapore")

    Returns:
        List of raw place dictionaries from Google Places API

    Examples:
        # Single type
        search_places(location, included_types="restaurant", excluded_types=["bar", "pub"], destination_city="Singapore")

        # Multiple types
        search_places(location, included_types=["restaurant", "cafe"], excluded_types=["bar"], destination_city="Singapore")
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

    # Handle both tuple and dict location formats
    if isinstance(location, dict):
        lat = location['lat']
        lng = location.get('lng') or location.get('lon')
    elif isinstance(location, (tuple, list)) and len(location) >= 2:
        lat, lng = location[0], location[1]
    else:
        raise ValueError("location must be (lat, lng) tuple or dict with 'lat'/'lng' keys")

    # Validate parameters
    if radius <= 0 or radius > 35000:
        raise ValueError("radius must be between 1 and 35000 meters")

    url = "https://places.googleapis.com/v1/places:searchNearby"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.types,places.primaryType,places.priceLevel,places.businessStatus,places.editorialSummary,places.websiteUri,places.regularOpeningHours,places.accessibilityOptions"
    }

    body = {
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lng
                },
                "radius": radius
            }
        },
        "languageCode": language,
        "maxResultCount": min(max_results, 20)
    }

    # Handle included_types - can be string or list
    if included_types:
        if isinstance(included_types, str):
            body["includedTypes"] = [included_types]
        else:
            body["includedTypes"] = included_types

    # Handle excluded_types - always a list
    if excluded_types:
        body["excludedTypes"] = excluded_types if isinstance(excluded_types, list) else [excluded_types]

    # Format for logging
    included_str = included_types if isinstance(included_types, str) else included_types
    excluded_str = f", excluded={excluded_types}" if excluded_types else ""
    print(f"Searching nearby: radius={radius}m, included={included_str}{excluded_str}")

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        data = response.json()

        places = data.get('places', [])
        print(f"Found {len(places)} places")

        # Filter by rating, businessStatus, and destination city address
        filtered_results = []
        for place in places:
            business_status = place.get('businessStatus', 'OPERATIONAL')
            if business_status != 'OPERATIONAL':
                print(f"Filtered non-operational: {place.get('displayName', {}).get('text', 'Unknown')}")
                continue

            rating = place.get('rating', 0)
            if rating < min_rating:
                continue

            # Filter: address must contain destination city (if specified)
            if destination_city:
                address = place.get('formattedAddress', '').lower()
                destination_lower = destination_city.lower()
                if destination_lower not in address:
                    place_name = place.get('displayName', {}).get('text', 'Unknown') if isinstance(place.get('displayName'), dict) else 'Unknown'
                    print(f"Filtered non-{destination_city}: {place_name} ({place.get('formattedAddress', 'No address')})")
                    continue

            # Filter: Remove hotels that are NOT tourist attractions
            # If types contains "hotel" but NOT "tourist_attraction", skip this place
            place_types = place.get('types', [])
            if place_types:
                has_hotel = 'hotel' in place_types
                has_tourist_attraction = 'tourist_attraction' in place_types

                if has_hotel and not has_tourist_attraction:
                    place_name = place.get('displayName', {}).get('text', 'Unknown') if isinstance(place.get('displayName'), dict) else 'Unknown'
                    print(f"Filtered non-tourist hotel: {place_name} (types: {place_types})")
                    continue

            filtered_results.append(place)

        print(f"Returning {len(filtered_results)} places after filtering")
        return filtered_results

    except requests.exceptions.HTTPError as e:
        # Print actual API error response
        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', str(e))
            print(f"[ERROR] API returned {response.status_code}: {error_msg}")
            logger.error(f"Search nearby HTTP {response.status_code}: {error_msg}")
        except:
            print(f"[ERROR] Search nearby: {e}")
            logger.error(f"Search nearby error: {e}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Search nearby request error: {e}")
        print(f"[ERROR] Request failed: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"[ERROR] Unexpected: {e}")
        return []

def geocode_location(location_name: str, country: str = "Singapore") -> Optional[Dict]:
    """
    Geocode a location name (neighbourhood, address, or place name) to lat/lng coordinates
    using Google Places API v1 Text Search.

    Args:
        location_name: Name of the location (e.g., "Clarke Quay", "Marina Bay", "Orchard Road")
        country: Country to search in (default: "Singapore")

    Returns:
        Dict with 'lat', 'lng', 'place_id', and 'place_name' keys if successful, None if geocoding fails

    Example:
        result = geocode_location("Clarke Quay")
        # Returns: {'lat': 1.2931, 'lng': 103.8467, 'place_id': 'ChIJ...', 'place_name': 'Clarke Quay'}
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

    # Use Text Search API for geocoding
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location"
    }

    # Format query to include country for better accuracy
    query = f"{location_name}, {country}" if country else location_name

    body = {
        "textQuery": query,
        "languageCode": "en",
        "maxResultCount": 1  # Only need the top result
    }

    print(f"Geocoding location: '{location_name}' in {country}...")

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()

        places = data.get('places', [])
        if not places:
            print(f"No results found for '{location_name}'")
            return None

        # Extract location from first result
        place = places[0]
        location = place.get('location', {})
        display_name = place.get('displayName', {})
        formatted_address = place.get('formattedAddress', '')
        place_id = place.get('id', '')

        # Extract place_id (API v1 format: "places/{place_id}")
        if place_id.startswith('places/'):
            place_id = place_id.replace('places/', '')

        lat = location.get('latitude')
        lng = location.get('longitude')

        if lat is not None and lng is not None:
            place_name = display_name.get('text', location_name) if isinstance(display_name, dict) else display_name
            print(f"[OK] Geocoded '{location_name}' to {place_name}: ({lat:.4f}, {lng:.4f})")
            print(f"  Place ID: {place_id}")
            print(f"  Address: {formatted_address}")
            return {'lat': lat, 'lng': lng, 'place_id': place_id, 'place_name': place_name}

        print(f"Location data incomplete for '{location_name}'")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Geocoding error for '{location_name}': {e}")
        print(f"[ERROR] Geocoding failed for '{location_name}': {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected geocoding error: {e}")
        print(f"[ERROR] Unexpected error geocoding '{location_name}': {e}")
        return None


def reverse_geocode(lat: float, lng: float) -> Optional[Dict]:
    """
    Reverse geocode coordinates to get place details using Google Places API v1.
    Prioritizes accommodation types (hotel, lodging, resort) over other place types.

    Args:
        lat: Latitude
        lng: Longitude

    Returns:
        Dict with 'place_id' and 'name' keys if successful, None if reverse geocoding fails

    Example:
        result = reverse_geocode(1.2931, 103.8467)
        # Returns: {'place_id': 'ChIJ...', 'name': 'Marina Bay Sands'}
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

    # Priority order: try accommodation types first, then expand to general places
    search_strategies = [
        {
            'types': ['hotel', 'lodging', 'resort_hotel'],
            'radius': 100.0,
            'description': 'accommodation within 100m'
        },
        {
            'types': ['hotel', 'lodging', 'resort_hotel'],
            'radius': 500.0,
            'description': 'accommodation within 500m'
        },
        {
            'types': ['shopping_mall'],
            'radius': 200.0,
            'description': 'shopping mall within 200m'
        },
        {
            'types': None,  # No type filter - find any nearby place
            'radius': 50.0,
            'description': 'any place within 50m'
        }
    ]

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.types"
    }

    print(f"Reverse geocoding coordinates: ({lat:.4f}, {lng:.4f})...")

    for strategy in search_strategies:
        body = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": strategy['radius']
                }
            },
            "maxResultCount": 5,  # Get top 5 to filter
            "rankPreference": "DISTANCE"
        }

        # Add type filter if specified
        if strategy['types']:
            body["includedTypes"] = strategy['types']

        try:
            response = requests.post(url, headers=headers, json=body, timeout=10)
            response.raise_for_status()
            data = response.json()

            places = data.get('places', [])
            if not places:
                print(f"  No {strategy['description']} found, trying next strategy...")
                continue

            # Extract details from first result
            place = places[0]
            display_name = place.get('displayName', {})
            place_id = place.get('id', '')
            place_types = place.get('types', [])

            # Extract place_id (API v1 format: "places/{place_id}")
            if place_id.startswith('places/'):
                place_id = place_id.replace('places/', '')

            place_name = display_name.get('text', '') if isinstance(display_name, dict) else display_name

            if place_id and place_name:
                place_type_str = ', '.join(place_types[:3]) if place_types else 'unknown type'
                print(f"[OK] Reverse geocoded to: {place_name} ({place_type_str})")
                print(f"  Place ID: {place_id}")
                return {'place_id': place_id, 'name': place_name}

        except requests.exceptions.RequestException as e:
            logger.error(f"Reverse geocoding error for ({lat:.4f}, {lng:.4f}): {e}")
            print(f"[ERROR] Reverse geocoding failed: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected reverse geocoding error: {e}")
            print(f"[ERROR] Unexpected error during reverse geocoding: {e}")
            continue

    print(f"No place found at coordinates ({lat:.4f}, {lng:.4f})")
    return None


def analyze_interests_with_llm(interests: List[str], openai_client) -> List[Dict]:
    """
    Use LLM to analyze interests and classify them as location-based or category-based.

    Location-based: "exploring near Changi Airport", "around Marina Bay"
    Category-based: "museums", "gardens", "adventure activities"

    Args:
        interests: List of interest strings from user input
        openai_client: OpenAI client for LLM calls

    Returns:
        List of dicts with structure:
        {
            "original": "exploring near Changi Airport",
            "type": "location",  # or "category"
            "location_query": "Changi Airport Singapore",  # if type is "location"
            "categories": []  # if type is "category", list of generic categories
        }
    """
    if not interests or not openai_client:
        return []

    print(f"\nAnalyzing {len(interests)} interest(s) with LLM...")

    # Create prompt for LLM
    prompt = f"""Analyze the following user interests and classify each as either "location" or "category".

Location-based interests mention specific places or areas (e.g., "exploring near Changi Airport", "around Marina Bay", "in Chinatown").
Category-based interests mention types of activities or places without specific locations (e.g., "museums", "gardens", "adventure", "food courts").

For location-based interests, extract the location name for searching.
For category-based interests, identify generic attraction categories.

User interests:
{chr(10).join(f'{i+1}. {interest}' for i, interest in enumerate(interests))}

Respond in JSON format as an array of objects:
[
    {{
        "original": "the exact original interest text",
        "type": "location" or "category",
        "location_query": "location name" (only if type is location),
        "categories": ["category1", "category2"] (only if type is category, use generic terms like "museum", "garden", "shopping")
    }}
]

Important:
- For location interests, extract ONLY the location name (e.g., "Changi Airport", "Marina Bay")
- For category interests, use generic attraction types
- Be precise with the "type" field
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a travel planning assistant that analyzes user interests."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        import json
        parsed = json.loads(result_text)

        # Handle both array and object with array responses
        if isinstance(parsed, dict) and 'interests' in parsed:
            analyzed = parsed['interests']
        elif isinstance(parsed, dict) and 'results' in parsed:
            analyzed = parsed['results']
        elif isinstance(parsed, list):
            analyzed = parsed
        else:
            analyzed = [parsed]

        # Log analysis results
        for item in analyzed:
            original = item.get('original', '')
            interest_type = item.get('type', 'unknown')
            if interest_type == 'location':
                location_query = item.get('location_query', '')
                print(f"  '{original}' -> LOCATION: {location_query}")
            else:
                categories = item.get('categories', [])
                print(f"  '{original}' -> CATEGORY: {', '.join(categories)}")

        return analyzed

    except Exception as e:
        logger.exception(f"Error analyzing interests with LLM: {e}")
        print(f"[ERROR] Failed to analyze interests: {e}")
        # Fallback: treat all as category-based
        return [{"original": interest, "type": "category", "categories": []} for interest in interests]


def get_place_details(place_ids, field=None, details_per_second=5.0, max_retries=3, language='en', max_workers=5):
    """
    Fetch detailed information for places using Google Places API v1 with concurrent processing.
    Uses ThreadPoolExecutor to fetch details concurrently while respecting rate limits.
    Returns RAW API data without formatting.

    Args:
        place_ids: List of place IDs to get details for
        field: List of fields to retrieve
        details_per_second: Rate limit for API calls (default: 5/sec)
        max_retries: Number of retries for failed requests
        language: Language for results
        max_workers: Maximum concurrent workers (default: 5)

    Returns:
        Dict mapping {place_id: raw_details_dict}
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

    details_by_id = {}
    sleep_between = 1.0 / details_per_second

    # Thread-safe lock for rate limiting
    rate_limit_lock = threading.Lock()
    last_request_time = {'time': 0}

    print(f"Getting details for {len(place_ids)} places (concurrent, rate limit {details_per_second}/sec)")

    def fetch_single_detail(place_id: str, index: int) -> tuple:
        """Fetch details for a single place with rate limiting."""

        # Rate limiting with thread-safe lock
        with rate_limit_lock:
            elapsed = time.time() - last_request_time['time']
            if elapsed < sleep_between:
                time.sleep(sleep_between - elapsed)
            last_request_time['time'] = time.time()

        url = f"https://places.googleapis.com/v1/{place_id}"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "id,displayName,formattedAddress,location,rating,userRatingCount,types,primaryType,priceLevel,businessStatus,editorialSummary,websiteUri,regularOpeningHours,accessibilityOptions"
        }

        params = {"languageCode": language}

        attempts = 0
        while attempts <= max_retries:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                result = response.json()
                print(f"  [{index}/{len(place_ids)}] OK {place_id}")
                return (place_id, result)

            except requests.exceptions.HTTPError as e:
                attempts += 1
                if attempts > max_retries:
                    logger.error(f"Details failed for {place_id} after {attempts} attempts: {e}")
                    print(f"  [{index}/{len(place_ids)}] FAILED {place_id}")
                    return (place_id, {})

                backoff = 0.5 * (2 ** (attempts - 1))
                logger.warning(f"HTTP error for {place_id}, backing off {backoff:.2f}s (attempt {attempts}): {e}")
                time.sleep(backoff)

            except Exception as e:
                logger.exception(f"Unexpected error for {place_id}: {e}")
                print(f"  [{index}/{len(place_ids)}] ERROR {place_id}")
                return (place_id, {})

        return (place_id, {})

    # Use ThreadPoolExecutor for concurrent fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_detail, place_id, i): place_id
        for i, place_id in enumerate(place_ids, 1)}

        for future in as_completed(futures):
            place_id, details = future.result()
            details_by_id[place_id] = details

    return details_by_id


def generate_tags(
    place_type: str,
    all_types: List[str],
    accessibility_options: List[str],
    price_level: Optional[int],
    rating: Optional[float],
    description: str,
    name: str,
    reviews_count: Optional[int] = None,
    openai_client=None
) -> List[str]:
    """
    Generate enhanced tags for a place based on available data.

    Uses a hybrid approach:
    1. Rule-based tags from place data (fast, deterministic)
    2. LLM-extracted tags from description (context-rich)

    Args:
        place_type: Primary mapped type
        all_types: All Google Place types
        accessibility_options: List of accessibility features
        price_level: Price level (0-4)
        rating: Place rating
        description: Place description (from Google)
        name: Place name
        reviews_count: Total number of reviews (user_ratings_total)
        openai_client: OpenAI client instance for LLM tag extraction

    Returns:
        List of descriptive tags
    """
    from config import (
        FOOD_TYPE_MAPPINGS,
        ATTRACTION_TYPE_MAPPINGS,
        SINGAPORE_AREA_MAPPINGS,
        TAG_LIMITS,
        RATING_TAG_THRESHOLDS
    )

    tags = []

    # 1. Always include primary type
    tags.append(place_type)

    # 2. Add wheelchair-friendly tag if accessible
    if accessibility_options and "wheelchair_accessible_entrance" in accessibility_options:
        tags.append("wheelchair-friendly")

    # 3. Extract food-specific tags from Google types
    for gtype in all_types:
        gtype_lower = gtype.lower()
        if gtype_lower in FOOD_TYPE_MAPPINGS:
            tags.append(FOOD_TYPE_MAPPINGS[gtype_lower])

    # 4. Add attraction-specific tags from Google types
    for gtype in all_types:
        gtype_lower = gtype.lower()
        if gtype_lower in ATTRACTION_TYPE_MAPPINGS:
            tag_value = ATTRACTION_TYPE_MAPPINGS[gtype_lower]
            if tag_value not in tags:
                tags.append(tag_value)

    # 5. Add price-based tags
    if price_level is not None:
        if price_level == 0:
            tags.append("free")
        elif price_level == 1:
            tags.append("budget-friendly")
        elif price_level in [3, 4]:
            tags.append("premium")

    # 6. Add rating-based tags (with review count requirements)
    if rating is not None:
        # Highly-rated requires both high rating AND sufficient reviews
        if (rating >= RATING_TAG_THRESHOLDS["highly_rated"]["min_rating"] and
            reviews_count is not None and
            reviews_count > RATING_TAG_THRESHOLDS["highly_rated"]["min_reviews"]):
            tags.append("highly-rated")
        # Top-rated only requires rating (no review requirement)
        elif rating >= RATING_TAG_THRESHOLDS["top_rated"]["min_rating"]:
            if reviews_count is not None and reviews_count > RATING_TAG_THRESHOLDS["top_rated"]["min_reviews"]:
                tags.append("top-rated")

    # 7. Add location-based tags from name (Singapore-specific)
    name_lower = name.lower()
    for area_name, area_tag in SINGAPORE_AREA_MAPPINGS.items():
        if area_name in name_lower:
            tags.append(area_tag)

    # 8. Use LLM to extract additional descriptive tags from description
    # Only if description is from Google (not generic fallback) and client is provided
    if openai_client and description and not description.startswith("A "):
        llm_tags = extract_tags_from_description(
            description, name, place_type, tags, openai_client
        )
        tags.extend(llm_tags)

    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    # Limit to max tags for readability
    return unique_tags[:TAG_LIMITS["max_total_tags"]]


def extract_tags_from_description(
    description: str,
    name: str,
    place_type: str,
    existing_tags: List[str],
    openai_client
) -> List[str]:
    """
    Use LLM to extract additional descriptive tags from place description.

    Args:
        description: Google description
        name: Place name
        place_type: Primary type
        existing_tags: Tags already generated
        openai_client: OpenAI client instance

    Returns:
        List of additional tags (max 3)
    """
    from config import (
        TAG_EXTRACTION_PROMPT,
        TAG_EXTRACTION_SYSTEM_PROMPT,
        LLM_CONFIG,
        TAG_LIMITS
    )
    import json
    import re
    import logging

    logger = logging.getLogger(__name__)

    try:
        prompt = TAG_EXTRACTION_PROMPT.format(
            name=name,
            place_type=place_type,
            description=description,
            existing_tags=', '.join(existing_tags)
        )

        response = openai_client.chat.completions.create(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            messages=[
                {"role": "system", "content": TAG_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=LLM_CONFIG["max_tokens_tags"]
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON array
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            extracted_tags = json.loads(json_match.group())
            # Filter out any tags that are already in existing_tags
            new_tags = [t.lower() for t in extracted_tags if isinstance(t, str) and t.lower() not in existing_tags]
            return new_tags[:TAG_LIMITS["max_llm_tags"]]  # Max 3 additional tags

        return []

    except Exception as e:
        logger.debug(f"LLM tag extraction failed for {name}: {e}")
        return []

def convert_dietary_to_exclusions(dietary_restrictions: List[str]) -> List[str]:
    """
    Convert dietary restrictions to Google Places API excluded types.

    Args:
        dietary_restrictions: List of dietary restrictions (e.g., ["vegetarian", "halal"])

    Returns:
        List of place types to exclude
    """
    excluded_types = []
    for restriction in dietary_restrictions:
        restriction_lower = restriction.lower().strip()
        if restriction_lower in DIETARY_EXCLUSIONS:
            excluded_types.extend(DIETARY_EXCLUSIONS[restriction_lower])

    # Deduplicate
    return list(set(excluded_types))


def generate_place_description(place_data: dict, openai_client=None) -> str:
    """
    Generate a Singapore-specific place description using LLM.
    Only call this if API doesn't provide editorialSummary.

    Args:
        place_data: Dictionary containing place information
        openai_client: OpenAI client instance (optional)

    Returns:
        Generated description string, or fallback if generation fails
    """
    from config import PLACE_DESCRIPTION_PROMPT, LLM_CONFIG
    import logging

    logger = logging.getLogger(__name__)

    # If no OpenAI client, return fallback
    if not openai_client:
        return f"A {place_data.get('type', 'place')} in Singapore."

    try:
        prompt = PLACE_DESCRIPTION_PROMPT.format(
            name=place_data.get('name', 'Unknown'),
            address=place_data.get('address', 'No address'),
            lat=place_data.get('latitude', 0),
            lng=place_data.get('longitude', 0),
            neighborhood=place_data.get('neighborhood', 'Unknown area'),
            place_type=place_data.get('type', 'Unknown')
        )

        response = openai_client.chat.completions.create(
            model=LLM_CONFIG["model"],
            temperature=LLM_CONFIG["temperature"],
            messages=[
                {"role": "system", "content": "You are a Singapore travel expert. Generate concise, informative place descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        logger.debug(f"LLM description generation failed for {place_data.get('name')}: {e}")
        return f"A {place_data.get('type', 'place')} located at {place_data.get('address', 'Singapore')}."


def map_interest_to_place_types(interest: str, openai_client) -> List[str]:
    """
    Map a user interest/category to valid Google Place types using LLM.

    This function uses ATTRACTION_PLACE_TYPES to ensure only valid Google types are returned.

    Args:
        interest: User interest or category (e.g., "museums", "temples", "nature")
        openai_client: OpenAI client instance for LLM calls

    Returns:
        List of 1-3 valid Google Place types (e.g., ["museum", "art_gallery"])
    """
    from config import ATTRACTION_PLACE_TYPES, LLM_CONFIG
    import json
    import re

    logger = logging.getLogger(__name__)

    # Convert set to sorted list for prompt
    valid_types_list = sorted(list(ATTRACTION_PLACE_TYPES))

    prompt = f"""Map the user interest "{interest}" to 1-3 most relevant Google Place types.

ONLY use types from this list (these are the ONLY valid Google Place types for attractions):
{', '.join(valid_types_list)}

Return ONLY a JSON array of 1-3 type strings. Example: ["museum", "art_gallery"]

Interest: {interest}
Types:"""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_CONFIG["model"],
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a type mapping assistant. Map user interests to valid Google Place types. Return ONLY a JSON array, nothing else."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON array
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            place_types = json.loads(json_match.group())
            # Validate types are in ATTRACTION_PLACE_TYPES
            valid_place_types = [t for t in place_types if isinstance(t, str) and t in ATTRACTION_PLACE_TYPES]
            logger.info(f"Mapped interest '{interest}' -> {valid_place_types}")
            return valid_place_types[:3]  # Max 3 types

        logger.warning(f"LLM mapping failed to return JSON for '{interest}'")
        return ["tourist_attraction"]  # Fallback

    except Exception as e:
        logger.warning(f"LLM mapping failed for interest '{interest}': {e}")
        return ["tourist_attraction"]  # Fallback
