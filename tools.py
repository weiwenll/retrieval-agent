import os
import googlemaps

def get_routing_matrix(origins, destinations, mode="train", units="metric", language="en", cutoff_times=None, avoid=None, departure_time=None):
    """
    Get travel times and distances between origins and destinations using Google Distance Matrix API.
    Filters results based on cutoff time limits for each transport mode.

    Args:
        origins: List of location dicts [{'lat': float, 'lng': float}] or address strings
        destinations: List of location dicts [{'lat': float, 'lng': float}] or address strings
        mode: Transport mode ('driving', 'walking', 'transit', 'bicycling')
        cutoff_times: Dict with time limits in minutes {'driving': 30, 'transit': 60, 'walking': 20}
        language: Language code for results

    Returns:
        Dict with filtered routing matrix data:
        {
            'mode': str,
            'filtered_routes': [
                {
                    'origin_index': int,
                    'destination_index': int,
                    'duration_minutes': int,
                    'distance_km': float,
                    'status': str
                }
            ],
            'total_routes': int,
            'filtered_count': int
        }
    """
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not set in environment")

    gmaps = googlemaps.Client(key=api_key)

    # Default cutoff times (in minutes)
    default_cutoffs = {
        'driving': 45,
        'walking': 15,
        'transit': 60,
        'bicycling': 20
    }

    if cutoff_times is None:
        cutoff_times = default_cutoffs
    else:
        # Merge with defaults for missing modes
        cutoff_times = {**default_cutoffs, **cutoff_times}

    # Convert location dicts to coordinate strings if needed
    def format_location(loc):
        if isinstance(loc, dict) and 'lat' in loc:
            lng = loc.get('lng') or loc.get('lon')
            return f"{loc['lat']},{lng}"
        return str(loc)  # Assume it's already an address string

    formatted_origins = [format_location(origin) for origin in origins]
    formatted_destinations = [format_location(dest) for dest in destinations]

    print(f"Getting {mode} routes from {len(origins)} origins to {len(destinations)} destinations...")

    try:
        # Call Google Distance Matrix API
        response = gmaps.distance_matrix(
            origins=formatted_origins,
            destinations=formatted_destinations,
            mode=mode,
            language=language,
            units='metric',
            avoid=avoid,
            departure_time=departure_time
        )

        # Process and filter results
        filtered_routes = []
        total_routes = 0
        cutoff_minutes = cutoff_times.get(mode, 60)

        rows = response.get('rows', [])
        for origin_idx, row in enumerate(rows):
            elements = row.get('elements', [])
            for dest_idx, element in enumerate(elements):
                total_routes += 1

                status = element.get('status', 'UNKNOWN')
                if status != 'OK':
                    continue

                # Extract duration and distance
                duration_data = element.get('duration', {})
                distance_data = element.get('distance', {})

                duration_seconds = duration_data.get('value', 0)
                duration_minutes = duration_seconds / 60

                distance_meters = distance_data.get('value', 0)
                distance_km = distance_meters / 1000

                # Apply cutoff filter
                if duration_minutes <= cutoff_minutes:
                    filtered_routes.append({
                        'origin_index': origin_idx,
                        'destination_index': dest_idx,
                        'duration_minutes': round(duration_minutes, 1),
                        'distance_km': round(distance_km, 2),
                        'duration_text': duration_data.get('text', ''),
                        'distance_text': distance_data.get('text', ''),
                        'status': status
                    })

        print(f"Found {len(filtered_routes)} routes under {cutoff_minutes} min cutoff (from {total_routes} total)")

        return {
            'mode': mode,
            'cutoff_minutes': cutoff_minutes,
            'filtered_routes': filtered_routes,
            'total_routes': total_routes,
            'filtered_count': len(filtered_routes),
            'origins_count': len(origins),
            'destinations_count': len(destinations)
        }

    except googlemaps.exceptions.ApiError as e:
        print(f"Distance Matrix API error: {e}")
        return {
            'mode': mode,
            'error': str(e),
            'filtered_routes': [],
            'total_routes': 0,
            'filtered_count': 0
        }
    except Exception as e:
        print(f"Unexpected error in routing matrix: {e}")
        return {
            'mode': mode,
            'error': str(e),
            'filtered_routes': [],
            'total_routes': 0,
            'filtered_count': 0
        }
