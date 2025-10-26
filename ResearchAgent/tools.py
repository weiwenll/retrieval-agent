import os
import time
import unicodedata
from typing import Optional, Dict, List
from venv import logger
import googlemaps
import wikipedia
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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

def search_places(location, radius=2000, keyword=None, language='en', max_pages=1, min_rating=0.0):
    """
    Search Google Places nearby with pagination support using the old/legacy API.
    Returns RAW API data without formatting.
    
    Args:
        location: (lat, lng) tuple or dict with 'lat'/'lng' keys  
        radius: search radius in meters (default 20km)
        keyword: optional search keyword
        language: language code (default 'en') 
        max_pages: maximum pages to fetch (1-3, each page = ~20 results)
        min_rating: minimum rating filter (0.0-5.0)
    
    Returns:
        List of raw place dictionaries from Google Places API
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Handle both tuple and dict location formats
    if isinstance(location, dict):
        # Handle both 'lng' and 'lon' keys
        lat = location['lat']
        lng = location.get('lng') or location.get('lon')
        location_tuple = (lat, lng)
    elif isinstance(location, (tuple, list)) and len(location) >= 2:
        location_tuple = (location[0], location[1])
    else:
        raise ValueError("location must be (lat, lng) tuple or dict with 'lat'/'lng' keys")
    
    # Validate parameters
    if not (1 <= max_pages <= 3):
        raise ValueError("max_pages must be between 1 and 3")
    if radius <= 0 or radius > 50000:
        raise ValueError("radius must be between 1 and 50000 meters")
    
    raw_results = []
    page_token = None
    pages_fetched = 0
    
    # Fetch pages with pagination
    while pages_fetched < max_pages:
        try:
            if page_token:
                print(f"Fetching page {pages_fetched + 1}...")
                time.sleep(2)  # Required delay for next_page_token
                response = gmaps.places_nearby(page_token=page_token, language=language)
            else:
                print("Fetching page 1...")
                response = gmaps.places_nearby(
                    location=location_tuple,
                    radius=radius,
                    keyword=keyword,
                    language=language
                )
        except googlemaps.exceptions.ApiError as e:
            logger.exception("Places API error: %s", e)
            break
        except Exception as e:
            logger.exception("Unexpected error in nearby search: %s", e)
            break
        
        # Collect raw results
        page_results = response.get('results', [])
        raw_results.extend(page_results)
        
        # Check for next page
        page_token = response.get('next_page_token')
        pages_fetched += 1
        
        if not page_token:
            print("No more pages available")
            break
    
    # Filter by rating - return raw data
    filtered_results = []
    for place in raw_results:
        # Check rating filter
        rating = place.get('rating', 0)
        if rating < min_rating:
            continue
        
        # Return the raw place data from Google API
        filtered_results.append(place)
    
    return filtered_results

def search_multiple_keywords(location, keywords, radius=2000, max_pages=1, min_rating=4.0):
    """
    Search for multiple keywords and return raw results with deduplication.
    Returns RAW API data without formatting.
    
    Args:
        location: (lat, lng) tuple or dict with 'lat'/'lng' keys
        keywords: list of keywords to search for
        radius: search radius in meters
        max_pages: max pages per keyword (to avoid too many API calls)
        min_rating: minimum rating filter
    
    Returns:
        List of unique raw places from Google API
    """
    all_results = []
    seen_place_ids: Set[str] = set()
    
    for keyword in keywords:
        print(f"\n=== Searching for: '{keyword}' ===")
        
        try:
            # Use search_places instead of search_nearby (which doesn't exist)
            results = search_places(
                location=location,
                radius=radius,
                keyword=keyword,
                max_pages=max_pages,
                min_rating=min_rating
            )
            
            # Deduplicate by place_id
            new_results = []
            for place in results:
                place_id = place.get('place_id')
                if place_id and place_id not in seen_place_ids:
                    seen_place_ids.add(place_id)
                    new_results.append(place)
            
            all_results.extend(new_results)
            print(f"Added {len(new_results)} new unique places for '{keyword}' (total: {len(all_results)})")
            
        except Exception as e:
            print(f"Error searching for '{keyword}': {e}")
            continue
    
    return all_results

def get_place_details(place_ids, fields=None, details_per_second=5.0, max_retries=3, language='en', max_workers=5):
    """
    Fetch detailed information for places using the old Places API with concurrent processing.
    Uses ThreadPoolExecutor to fetch details concurrently while respecting rate limits.
    Returns RAW API data without formatting.

    Args:
        place_ids: List of place IDs to get details for
        fields: List of fields to retrieve (None for all basic fields)
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

    # Common fields for place details
    if fields is None:
        fields = [
            'name', 'formatted_address', 'geometry', 'opening_hours',
            'rating', 'user_ratings_total', 'reviews', 'website',
            'price_level', 'type', 'wheelchair_accessible_entrance'
        ]

    details_by_id = {}
    sleep_between = 1.0 / details_per_second

    # Thread-safe lock for rate limiting
    rate_limit_lock = threading.Lock()
    last_request_time = {'time': 0}

    print(f"Getting details for {len(place_ids)} places (concurrent with rate limit {details_per_second}/sec)...")

    def fetch_single_detail(place_id: str, index: int) -> tuple:
        """Fetch details for a single place with rate limiting."""
        gmaps = googlemaps.Client(key=api_key)  # Thread-safe client instance

        # Rate limiting with thread-safe lock
        with rate_limit_lock:
            elapsed = time.time() - last_request_time['time']
            if elapsed < sleep_between:
                time.sleep(sleep_between - elapsed)
            last_request_time['time'] = time.time()

        attempts = 0
        while attempts <= max_retries:
            try:
                response = gmaps.place(
                    place_id=place_id,
                    fields=fields,
                    language=language
                )
                result = response.get('result', {})
                print(f"  [{index}/{len(place_ids)}] OK {place_id}")
                return (place_id, result)

            except googlemaps.exceptions.ApiError as e:
                attempts += 1
                if attempts > max_retries:
                    logger.error("Details failed for %s after %d attempts: %s", place_id, attempts, e)
                    print(f"  [{index}/{len(place_ids)}] FAILED {place_id} (failed)")
                    return (place_id, {})

                backoff = 0.5 * (2 ** (attempts - 1))
                logger.warning("API error for %s, backing off %.2fs (attempt %d): %s", place_id, backoff, attempts, e)
                time.sleep(backoff)

            except Exception as e:
                logger.exception("Unexpected error for %s: %s", place_id, e)
                print(f"  [{index}/{len(place_ids)}] ERROR {place_id} (error)")
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

def search_wikipedia(search_term: str) -> Optional[str]:
    """
    Search Wikipedia and return the summary of the best match.
    Thread-safe for concurrent execution.
    """
    try:
        results = wikipedia.search(search_term)
        if not results:
            return None

        first_result = results[0]
        return wikipedia.summary(first_result, auto_suggest=False)

    except Exception as e:
        print(f"Wikipedia search error for '{search_term}': {e}")
        return None


def fetch_place_enrichments(places: List[Dict], fetch_details: bool = True, fetch_wikipedia: bool = True, max_workers_wiki: int = 10) -> Dict:
    """
    Fetch place details and Wikipedia data concurrently for a list of places.
    This is a wrapper that coordinates both enrichment types.

    Args:
        places: List of place dictionaries with 'place_id' and 'name' keys
        fetch_details: Whether to fetch Google Place details (rate-limited to 5/sec, handled internally)
        fetch_wikipedia: Whether to fetch Wikipedia summaries (concurrent, no rate limit)
        max_workers_wiki: Maximum concurrent Wikipedia fetches (default: 10)

    Returns:
        Dict with 'details' and 'wikipedia' keys containing the fetched data
        {
            'details': {place_id: details_dict, ...},
            'wikipedia': {place_name: summary_string, ...}
        }
    """
    result = {
        'details': {},
        'wikipedia': {}
    }

    if not places:
        return result

    print(f"\n=== Fetching enrichments for {len(places)} places ===")

    # Prepare data for concurrent fetching
    place_ids = [p.get('place_id') for p in places if p.get('place_id')]
    place_names = [p.get('name') for p in places if p.get('name')]

    # Use ThreadPoolExecutor to fetch both types concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        # Submit place details job (internally rate-limited)
        if fetch_details and place_ids:
            futures['details'] = executor.submit(get_place_details, place_ids)

        # Submit Wikipedia job (concurrent)
        if fetch_wikipedia and place_names:
            futures['wikipedia'] = executor.submit(_fetch_wikipedia_batch, place_names, max_workers_wiki)

        # Collect results
        if 'details' in futures:
            result['details'] = futures['details'].result()

        if 'wikipedia' in futures:
            result['wikipedia'] = futures['wikipedia'].result()

    return result


def _fetch_wikipedia_batch(place_names: List[str], max_workers: int = 10) -> Dict[str, Optional[str]]:
    """
    Fetch Wikipedia summaries concurrently for multiple places.

    Args:
        place_names: List of place names to search
        max_workers: Maximum concurrent Wikipedia fetches

    Returns:
        Dict mapping {place_name: wikipedia_summary}
    """
    wikipedia_data = {}

    print(f"Fetching Wikipedia data for {len(place_names)} places (concurrent, max {max_workers} workers)...")

    def fetch_single_wikipedia(name: str, index: int) -> tuple:
        """Helper to fetch Wikipedia for a single place."""
        try:
            summary = search_wikipedia(name)
            if summary:
                print(f"  [{index}/{len(place_names)}] OK Wikipedia: {name}")
            else:
                print(f"  [{index}/{len(place_names)}] NOTFOUND Wikipedia: {name} (not found)")
            return (name, summary)
        except Exception as e:
            print(f"  [{index}/{len(place_names)}] ERROR Wikipedia: {name} (error: {e})")
            return (name, None)

    # Use ThreadPoolExecutor for concurrent Wikipedia fetches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_wikipedia, name, i): name
                   for i, name in enumerate(place_names, 1)}

        for future in as_completed(futures):
            name, summary = future.result()
            wikipedia_data[name] = summary

    return wikipedia_data


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
        description: Place description (from Wikipedia)
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
    # Only if description is from Wikipedia (not generic fallback) and client is provided
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
        description: Wikipedia description
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
