"""
Configuration file for Transport Agent
Contains API keys, constants, and transport filtering thresholds
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Routes API Configuration
GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_ROUTES_API_VERSION = "v2"

# Transport Mode Mappings for Google Routes API
GOOGLE_TRAVEL_MODES = {
    "DRIVE": "ride",  # Ride-hailing (Grab/Private Hire/Taxi)
    "TRANSIT": "public_transport",
    "WALK": "walk"
}

# Transport Filtering Thresholds
TRANSPORT_THRESHOLDS = {
    "walk": {
        "api_max_distance_km": 10.0,  # Cap API queries at 10km max
        "api_max_duration_minutes": 240,  # Cap API queries at 240 minutes (4 hours)
        "convert_to_cycle_distance_km": 2.0,  # Convert to cycle if >2km
        "convert_to_cycle_duration_minutes": 20,  # Convert to cycle if >20 minutes
        "offer_both_threshold_km": 5.0,  # Offer both walk and cycle if distance <=5km
        "description": "Walking distance thresholds with cycling conversion logic"
    },
    "public_transport": {
        "max_transfers": 3,
        "max_duration_minutes": 60,
        "max_walking_portion_km": 1.5,
        "description": "Public transport practicality thresholds"
    },
    "ride": {
        "expensive_threshold_sgd": 30,
        "description": "Ride-hailing (Grab/taxi) cost threshold for flagging expensive trips"
    },
    "cycle": {
        "max_distance_km": 8.0,  # Cap at 8km whether from API or converted from walking
        "max_duration_minutes": 45,
        "description": "Cycling distance threshold"
    }
}

# Singapore specific configurations
SINGAPORE_CONFIG = {
    "currency": "SGD",
    "timezone": "Asia/Singapore",
    "country_code": "SG",
    # Average costs in SGD
    "transport_costs": {
        "mrt_base_fare": 0.92,
        "mrt_per_km": 0.12,
        "bus_base_fare": 0.83,
        "bus_per_km": 0.11,
        "taxi_flag_down": 4.00,
        "taxi_per_km": 0.55,
        "taxi_per_minute_waiting": 0.30
    }
}

# Concurrent API call configuration
CONCURRENT_CONFIG = {
    "max_workers": 4,  # Maximum concurrent API calls (reduced to avoid rate limits)
    "timeout_seconds": 30,  # Timeout for each API call
    "retry_attempts": 3,  # Number of retries for failed calls
    "retry_delay_seconds": 1  # Delay between retries
}


# Geo cluster IDs used in Singapore
GEO_CLUSTERS = ["central", "north", "south", "east", "west"]

def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are present.

    Returns:
        Dict mapping API key names to their availability status
    """
    return {
        "GOOGLE_MAPS_API_KEY": bool(GOOGLE_MAPS_API_KEY),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY)
    }

def get_transport_threshold(mode: str, threshold_type: str) -> Any:
    """
    Get specific threshold value for a transport mode.

    Args:
        mode: Transport mode (walk, public_transport, ride, cycle)
        threshold_type: Type of threshold (api_max_distance_km, max_duration_minutes, etc.)

    Returns:
        Threshold value or None if not found
    """
    if mode in TRANSPORT_THRESHOLDS:
        return TRANSPORT_THRESHOLDS[mode].get(threshold_type)
    return None

def estimate_taxi_cost(distance_km: float, duration_minutes: float) -> float:
    """
    Estimate taxi cost in SGD based on distance and duration.

    Args:
        distance_km: Distance in kilometers
        duration_minutes: Duration in minutes

    Returns:
        Estimated cost in SGD
    """
    costs = SINGAPORE_CONFIG["transport_costs"]

    # Base fare
    cost = costs["taxi_flag_down"]

    # Distance cost
    cost += distance_km * costs["taxi_per_km"]

    # Waiting time cost (assume 30% of duration is waiting/slow traffic)
    waiting_minutes = duration_minutes * 0.3
    cost += waiting_minutes * costs["taxi_per_minute_waiting"]

    return round(cost, 2)

def estimate_public_transport_cost(distance_km: float) -> Dict[str, float]:
    """
    Estimate public transport costs in SGD.

    Args:
        distance_km: Distance in kilometers

    Returns:
        Dict with estimated costs for MRT and bus
    """
    costs = SINGAPORE_CONFIG["transport_costs"]

    mrt_cost = costs["mrt_base_fare"] + (distance_km * costs["mrt_per_km"])
    bus_cost = costs["bus_base_fare"] + (distance_km * costs["bus_per_km"])

    return {
        "mrt": round(mrt_cost, 2),
        "bus": round(bus_cost, 2)
    }
