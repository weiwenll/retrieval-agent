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
