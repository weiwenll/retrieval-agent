import os
import time
from venv import logger
import googlemaps

def search_places(location, radius=20000, keyword=None, language='en', max_pages=3, min_rating=0.0):
    """
    Search Google Places nearby with pagination support using the old/legacy API.
    
    Args:
        location: (lat, lng) tuple or dict with 'lat'/'lng' keys  
        radius: search radius in meters (default 20km)
        keyword: optional search keyword
        language: language code (default 'en') 
        max_pages: maximum pages to fetch (1-3, each page = ~20 results)
        min_rating: minimum rating filter (0.0-5.0)
    
    Returns:
        List of formatted place dictionaries (up to 60 results)
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
    
    all_results = []
    raw_results = []
    page_token = None
    pages_fetched = 0
    
    # print(f"Searching near {location_tuple} with radius {radius}m")
    # if keyword:
    #     print(f"Keyword: '{keyword}'")
    
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
        
        # print(f"Page {pages_fetched}: Got {len(page_results)} results")
        
        if not page_token:
            print("No more pages available")
            break
    
    # Filter by rating and format results
    for place in raw_results:
        # Check rating filter
        rating = place.get('rating', 0)
        if rating < min_rating:
            continue
        
        # Extract location data
        geometry = place.get('geometry', {})
        location_data = geometry.get('location', {})
        
        # Extract opening hours (complex format, set to null for now)
        opening_hours = None
        if place.get('opening_hours'):
            opening_hours = {
                "monday": None,
                "tuesday": None,
                "wednesday": None,
                "thursday": None,
                "friday": None,
                "saturday": None,
                "sunday": None
            }
        
        # Get types and primary type
        types = place.get('types', [])
        primary_type = types[0] if types else None
        
        formatted_place = {
            "place_id": place.get('place_id'),
            "name": place.get('name'),
            "type": primary_type,
            "cost_sgd": place.get('price_level'),  # Google uses 0-4 scale
            "onsite_co2_kg": None,  # Not available from basic API
            "geo": {
                "latitude": location_data.get('lat'),
                "longitude": location_data.get('lng')
            } if location_data else None,
            "geo_cluster_id": None,  # Would need custom logic
            "address": place.get('formatted_address') or place.get('vicinity'),
            "nearest_mrt": None,  # Would need additional API calls
            "opening_hours": opening_hours,
            "duration_recommended_minutes": None,  # Not available from basic API
            "ticket_price_sgd": {
                "adult": None,
                "child": None,
                "senior": None
            },
            "vegetarian_friendly": None,  # Would need details API call
            "low_carbon_score": None,  # Custom metric
            "description": None,  # Would need details API call
            "links": {
                "official": None,  # Would need details API call
                "reviews": f"https://www.google.com/maps/place/?q=place_id:{place.get('place_id')}" if place.get('place_id') else None
            },
            "tags": types,  # Use Google's types as tags
            "rating": rating,  # Include rating for reference
            "user_ratings_total": place.get('user_ratings_total')
        }
        all_results.append(formatted_place)
    
    return all_results

def search_multiple_keywords(location, keywords, radius=20000, max_pages=2, min_rating=4.0):
    """
    Search for multiple keywords and return formatted results with deduplication.
    
    Args:
        location: (lat, lng) tuple or dict with 'lat'/'lng' keys
        keywords: list of keywords to search for
        radius: search radius in meters
        max_pages: max pages per keyword (to avoid too many API calls)
        min_rating: minimum rating filter
    
    Returns:
        List of unique formatted places
    """
    all_results = []
    seen_place_ids: Set[str] = set()
    
    for keyword in keywords:
        print(f"\n=== Searching for: '{keyword}' ===")
        
        try:
            results = search_nearby(
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
    
    Args:
        place_ids: List of place IDs to get details for
        fields: List of fields to retrieve (None for all basic fields)
        details_per_second: Rate limit for API calls
        max_retries: Number of retries for failed requests
        language: Language for results
    
    Returns:
        Dict mapping {place_id: details_dict}
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")
    
    gmaps = googlemaps.Client(key=api_key)
    
    # Common fields for place details
    if fields is None:
        fields = [
            'name', 'formatted_address', 'geometry', 'opening_hours',
            'rating', 'website', 'price_level', 'type', 'url'
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