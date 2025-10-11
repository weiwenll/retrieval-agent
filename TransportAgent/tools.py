"""
Enhanced Transport Tools for Google Routes API Integration
- TWO_WHEELER not used (motorbikes not relevant for tourists)
- Converts walking >2km to cycling category
- Creates transit route summaries
"""

import requests
import time
import logging
import threading
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    GOOGLE_MAPS_API_KEY,
    GOOGLE_ROUTES_API_URL,
    CONCURRENT_CONFIG,
    TRANSPORT_THRESHOLDS,
    estimate_taxi_cost,
    EMISSION_FACTORS
)

logger = logging.getLogger(__name__)

# Rate limiting for API calls
_rate_limit_lock = threading.Lock()
_last_request_time = {"time": 0}
_min_delay_between_requests = 0.5  # 500ms between requests

# Store raw Google Maps responses for debugging
_raw_responses = []


def compute_route(
    origin: Dict[str, float],
    destination: Dict[str, float],
    travel_mode: str,
    language: str = "en"
) -> Optional[Dict[str, Any]]:
    """
    Call Google Routes API v2 to compute route between two locations.

    Args:
        origin: Dict with 'latitude' and 'longitude' keys
        destination: Dict with 'latitude' and 'longitude' keys
        travel_mode: One of 'DRIVE', 'TRANSIT', 'WALK'
        language: Language code for response

    Returns:
        Route data dict or None if failed
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.error("GOOGLE_MAPS_API_KEY not set")
        return None

    # Prepare request body
    request_body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin["latitude"],
                    "longitude": origin["longitude"]
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination["latitude"],
                    "longitude": destination["longitude"]
                }
            }
        },
        "travelMode": travel_mode,
        "computeAlternativeRoutes": False,
        "languageCode": language,
        "units": "METRIC"
    }

    # Only add routing preference for DRIVE mode (not allowed for WALK, TRANSIT)
    if travel_mode == "DRIVE":
        request_body["routingPreference"] = "TRAFFIC_AWARE"
        request_body["routeModifiers"] = {
            "avoidTolls": False,
            "avoidHighways": False,
            "avoidFerries": False
        }

    # Set headers
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.legs,routes.polyline"
    }

    # Apply rate limiting
    with _rate_limit_lock:
        elapsed = time.time() - _last_request_time["time"]
        if elapsed < _min_delay_between_requests:
            sleep_time = _min_delay_between_requests - elapsed
            time.sleep(sleep_time)
        _last_request_time["time"] = time.time()

    try:
        logger.info(f"Making API request for {travel_mode} mode")
        response = requests.post(
            GOOGLE_ROUTES_API_URL,
            json=request_body,
            headers=headers,
            timeout=CONCURRENT_CONFIG["timeout_seconds"]
        )

        logger.info(f"API response status: {response.status_code} for {travel_mode}")

        if response.status_code == 200:
            data = response.json()

            # Store raw response for debugging
            _raw_responses.append({
                "travel_mode": travel_mode,
                "origin": origin,
                "destination": destination,
                "response": data,
                "timestamp": time.time()
            })

            if "routes" in data and len(data["routes"]) > 0:
                logger.info(f"Successfully retrieved route for {travel_mode}")
                return data["routes"][0]
            else:
                logger.warning(f"No routes found for {travel_mode}. Response: {data}")
                return None
        else:
            logger.error(f"Routes API error {response.status_code} for {travel_mode}")
            logger.error(f"Response body: {response.text[:500]}")  # First 500 chars
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception for {travel_mode}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {travel_mode}: {e}")
        return None


def create_transit_summary(transit_steps: List[Dict]) -> str:
    """
    Create human-readable transit summary from transit steps.

    Example: "Take MRT Red Line to Raffles, then Bus 174 to destination"

    Args:
        transit_steps: List of transit step dicts with 'line' and 'vehicle'

    Returns:
        Human-readable summary string
    """
    if not transit_steps:
        return "Direct public transport"

    summary_parts = []
    for i, step in enumerate(transit_steps):
        line_name = step.get("line", "Unknown")
        vehicle_type = step.get("vehicle", "Unknown").lower()

        # Simplify vehicle type names
        if "subway" in vehicle_type or "metro" in vehicle_type or "mrt" in vehicle_type.lower():
            vehicle = "MRT"
        elif "bus" in vehicle_type:
            vehicle = "Bus"
        elif "train" in vehicle_type:
            vehicle = "Train"
        else:
            vehicle = vehicle_type.capitalize()

        if i == 0:
            summary_parts.append(f"Take {vehicle} {line_name}")
        else:
            summary_parts.append(f"then {vehicle} {line_name}")

    return ", ".join(summary_parts)


def parse_route_data(route: Dict[str, Any], travel_mode: str) -> Dict[str, Any]:
    """
    Parse Google Routes API response into standardized format.

    - Creates cycling category for walking >2km
    - Adds transit summary
    - Includes raw transit steps for reference

    Args:
        route: Route data from API
        travel_mode: Travel mode used

    Returns:
        Parsed route data dict
    """
    if not route:
        return None

    try:
        # Extract basic info
        distance_meters = route.get("distanceMeters", 0)
        distance_km = distance_meters / 1000.0

        logger.info(f"Parsing {travel_mode}: distance_meters={distance_meters}, distance_km={distance_km}")

        # Parse duration (format: "123s")
        duration_str = route.get("duration", "0s")
        duration_seconds = int(duration_str.rstrip("s"))
        duration_minutes = duration_seconds / 60.0

        logger.info(f"Parsing {travel_mode}: duration={duration_seconds}s ({duration_minutes}min)")

        # Get legs info (contains steps, transfers, etc.)
        legs = route.get("legs", [])

        # Count transfers for transit
        num_transfers = 0
        transit_steps = []
        walking_distance_m = 0

        if legs:
            for leg in legs:
                steps = leg.get("steps", [])
                for step in steps:
                    travel_mode_step = step.get("travelMode", "")
                    if travel_mode_step == "TRANSIT":
                        transit_detail = step.get("transitDetails", {})
                        transit_steps.append({
                            "line": transit_detail.get("transitLine", {}).get("name", "Unknown"),
                            "vehicle": transit_detail.get("transitLine", {}).get("vehicle", {}).get("type", "Unknown")
                        })
                    elif travel_mode_step == "WALK":
                        walking_distance_m += step.get("distanceMeters", 0)

            # Transfers = number of transit segments - 1
            if len(transit_steps) > 0:
                num_transfers = len(transit_steps) - 1

        result = {
            "travel_mode": travel_mode,
            "distance_km": round(distance_km, 2),
            "distance_meters": distance_meters,
            "duration_minutes": round(duration_minutes, 1),
            "duration_seconds": duration_seconds,
        }

        # Add transit-specific data with summary
        if travel_mode == "TRANSIT":
            result["num_transfers"] = num_transfers
            result["transit_steps"] = transit_steps
            result["transit_summary"] = create_transit_summary(transit_steps)
            result["walking_distance_km"] = round(walking_distance_m / 1000.0, 2)

        # Estimate cost
        if travel_mode == "DRIVE":
            result["estimated_cost_sgd"] = estimate_taxi_cost(distance_km, duration_minutes)
        elif travel_mode == "TRANSIT":
            # Use simple MRT estimate
            result["estimated_cost_sgd"] = round(0.92 + (distance_km * 0.12), 2)
        else:
            result["estimated_cost_sgd"] = 0.0

        return result

    except Exception as e:
        logger.error(f"Error parsing route data: {e}")
        return None


def convert_walking_to_cycling(walking_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert walking route to cycling estimate for distances > 2km.

    Cycling assumptions:
    - Average speed: 15 km/h (leisurely pace)
    - Duration = distance / speed
    - No cost

    Args:
        walking_data: Walking route data

    Returns:
        Cycling route data (estimated, not from API)
    """
    distance_km = walking_data.get("distance_km", 0)

    # Estimate cycling duration (15 km/h average)
    cycling_speed_kmh = 15.0
    duration_hours = distance_km / cycling_speed_kmh
    duration_minutes = duration_hours * 60

    return {
        "travel_mode": "CYCLING",
        "distance_km": distance_km,
        "distance_meters": walking_data.get("distance_meters", 0),
        "duration_minutes": round(duration_minutes, 1),
        "duration_seconds": int(duration_minutes * 60),
        "estimated_cost_sgd": 0.0,
        "note": "Estimated based on walking route (not from Google API)"
    }


def get_transport_options_concurrent(
    origin: Dict[str, float],
    destination: Dict[str, float],
    modes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get transport options for all modes concurrently (parallel API calls).

    - Walking >2km converted to cycling
    - Walking <=5km still shown

    Args:
        origin: Dict with 'latitude' and 'longitude' keys
        destination: Dict with 'latitude' and 'longitude' keys
        modes: List of modes to check (default: DRIVE, TRANSIT, WALK)

    Returns:
        Dict mapping mode names to route data
    """
    if modes is None:
        modes = ["DRIVE", "TRANSIT", "WALK"]  # TWO_WHEELER removed

    results = {}

    def fetch_mode(mode: str) -> Tuple[str, Optional[Dict]]:
        """Helper function to fetch a single mode."""
        route = compute_route(origin, destination, mode)
        parsed = parse_route_data(route, mode)
        return (mode, parsed)

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=min(len(modes), CONCURRENT_CONFIG["max_workers"])) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_mode, mode): mode for mode in modes}

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                mode, parsed = future.result()
                if parsed:
                    results[mode] = parsed
                    logger.info(f"Successfully fetched route for {mode}")
                else:
                    logger.warning(f"No route data for {mode}")
            except Exception as e:
                mode = futures[future]
                logger.error(f"Error fetching route for {mode}: {e}")

    # Process walking results: convert >2km to cycling, keep <=5km as walking
    if "WALK" in results:
        walking_data = results["WALK"]
        distance_km = walking_data.get("distance_km", 0)

        if distance_km > 2.0:
            # Create cycling option
            cycling_data = convert_walking_to_cycling(walking_data)
            results["CYCLING"] = cycling_data
            logger.info(f"Created cycling option for {distance_km}km route")

            # Only keep walking if <=5km
            if distance_km > 5.0:
                logger.info(f"Removing walking option ({distance_km}km > 5km threshold)")
                del results["WALK"]

    return results


def batch_process_routes(
    location_pairs: List[Tuple[Dict, Dict, str, str]],
    concurrent: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple route requests efficiently.

    Args:
        location_pairs: List of tuples (origin, destination, origin_name, dest_name)
        concurrent: Whether to use concurrent processing

    Returns:
        List of dicts with route data for each pair
    """
    results = []

    def process_single_pair(pair_data: Tuple) -> Dict[str, Any]:
        """Process a single location pair."""
        origin, destination, origin_name, dest_name = pair_data

        logger.info(f"Processing route: {origin_name} -> {dest_name}")

        transport_options = get_transport_options_concurrent(origin, destination)

        return {
            "origin": origin_name,
            "destination": dest_name,
            "origin_coords": origin,
            "destination_coords": destination,
            "transport_options": transport_options
        }

    # Process all pairs
    if concurrent and len(location_pairs) > 1:
        # Use ThreadPoolExecutor for batch processing
        with ThreadPoolExecutor(max_workers=CONCURRENT_CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_single_pair, pair) for pair in location_pairs]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing pair: {e}")
    else:
        # Sequential processing
        for pair in location_pairs:
            try:
                result = process_single_pair(pair)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing pair: {e}")

    return results


def dump_raw_responses(output_file: str = "TransportAgent/transport_response.json"):
    """
    Dump all raw Google Maps API responses to JSON file for debugging.

    Args:
        output_file: Path to output file
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "total_requests": len(_raw_responses),
                "responses": _raw_responses
            }, f, indent=2)
        logger.info(f"Dumped {len(_raw_responses)} raw Google Maps responses to {output_file}")
    except Exception as e:
        logger.error(f"Failed to dump raw responses: {e}")


def clear_raw_responses():
    """Clear stored raw responses (call at start of new run)"""
    global _raw_responses
    _raw_responses = []


def carbon_estimate(mode: str, distance_km: float) -> float:
    """
    Estimate carbon emissions for a given transport mode and distance.

    Args:
        mode: Transport mode (taxi, bus, mrt, cycle, walking, driving)
        distance_km: Distance traveled in kilometers

    Returns:
        Carbon emissions in kg CO2
    """
    return EMISSION_FACTORS.get(mode, 0.05) * distance_km


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt

    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r
