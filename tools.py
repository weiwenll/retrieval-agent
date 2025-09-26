import os
import time
from venv import logger
import googlemaps

def search_places(location, radius=20000, keyword=None, language='en', max_pages=3, min_rating=0.0):
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

def search_multiple_keywords(location, keywords, radius=20000, max_pages=2, min_rating=4.0):
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

def get_place_details(place_ids, fields=None, details_per_second=5.0, max_retries=3, language='en'):
    """
    Fetch detailed information for places using the old Places API.
    Returns RAW API data without formatting.
    
    Args:
        place_ids: List of place IDs to get details for
        fields: List of fields to retrieve (None for all basic fields)
        details_per_second: Rate limit for API calls
        max_retries: Number of retries for failed requests
        language: Language for results
    
    Returns:
        Dict mapping {place_id: raw_details_dict}
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Common fields for place details
    if fields is None:
        fields = [
            'name', 'formatted_address', 'geometry', 'opening_hours',
            'rating', 'website', 'price_level', 'type'
        ]
    
    details_by_id = {}
    sleep_between = 1.0 / details_per_second
    
    print(f"Getting details for {len(place_ids)} places...")
    
    for i, place_id in enumerate(place_ids, 1):
        print(f"Fetching details {i}/{len(place_ids)}: {place_id}")
        
        attempts = 0
        while attempts <= max_retries:
            try:
                response = gmaps.place(
                    place_id=place_id,
                    fields=fields,
                    language=language
                )
                # Return raw response from API
                details_by_id[place_id] = response.get('result', {})
                break  # Success
                
            except googlemaps.exceptions.ApiError as e:
                attempts += 1
                if attempts > max_retries:
                    logger.error("Details failed for %s after %d attempts: %s", place_id, attempts, e)
                    details_by_id[place_id] = {}
                    break
                
                backoff = 0.5 * (2 ** (attempts - 1))
                logger.warning("API error for %s, backing off %.2fs (attempt %d): %s", place_id, backoff, attempts, e)
                time.sleep(backoff)
                
            except Exception as e:
                logger.exception("Unexpected error for %s: %s", place_id, e)
                details_by_id[place_id] = {}
                break
        
        # Rate limiting
        time.sleep(sleep_between)
    
    return details_by_id