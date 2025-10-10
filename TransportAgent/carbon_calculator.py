"""
Carbon Calculator for Transport Modes
Integrates with CLIMATIQ API to calculate carbon emissions
"""

import requests
import logging
from typing import Dict, Optional, Any
from config import (
    CLIMATIQ_API_KEY,
    CLIMATIQ_API_URL,
    CLIMATIQ_TRANSPORT_TYPES,
    FALLBACK_EMISSION_FACTORS
)

logger = logging.getLogger(__name__)


def calculate_carbon_emission(
    transport_mode: str,
    distance_km: float,
    use_fallback: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Calculate carbon emissions for a transport mode using CLIMATIQ API.

    Args:
        transport_mode: Transport mode (car, taxi, bus, train, mrt, bicycle, walking)
        distance_km: Distance traveled in kilometers
        use_fallback: Whether to use fallback calculation if API fails

    Returns:
        Dict with carbon emission data or None if failed
    """
    # Handle zero-emission modes
    if transport_mode in ["bicycle", "walking", "TWO_WHEELER", "WALK"]:
        return {
            "transport_mode": transport_mode,
            "distance_km": distance_km,
            "co2e_kg": 0.0,
            "co2e_source": "zero_emission",
            "activity_id": "none"
        }

    # Map Google mode names to our standard names
    mode_mapping = {
        "DRIVE": "car",
        "TRANSIT": "train",  # Will use MRT/train for Singapore
        "WALK": "walking",
        "TWO_WHEELER": "bicycle"
    }

    standard_mode = mode_mapping.get(transport_mode, transport_mode)

    # Try CLIMATIQ API first
    if CLIMATIQ_API_KEY and not use_fallback:
        try:
            result = _call_climatiq_api(standard_mode, distance_km)
            if result:
                return result
            else:
                logger.warning(f"CLIMATIQ API returned no data for {standard_mode}, using fallback")
        except Exception as e:
            logger.error(f"CLIMATIQ API error for {standard_mode}: {e}, using fallback")

    # Use fallback calculation
    return _calculate_fallback_emission(standard_mode, distance_km)


def _call_climatiq_api(transport_mode: str, distance_km: float) -> Optional[Dict[str, Any]]:
    """
    Call CLIMATIQ API to get carbon emissions.

    Args:
        transport_mode: Transport mode
        distance_km: Distance in kilometers

    Returns:
        Carbon emission data or None if failed
    """
    if not CLIMATIQ_API_KEY:
        logger.warning("CLIMATIQ_API_KEY not set")
        return None

    # Get activity ID for the transport mode
    activity_id = CLIMATIQ_TRANSPORT_TYPES.get(transport_mode)
    if not activity_id or activity_id == "none":
        return None

    # Prepare request
    headers = {
        "Authorization": f"Bearer {CLIMATIQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "emission_factor": {
            "activity_id": activity_id,
            "data_version": "^1"
        },
        "parameters": {
            "distance": distance_km,
            "distance_unit": "km"
        }
    }

    try:
        response = requests.post(
            CLIMATIQ_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            return {
                "transport_mode": transport_mode,
                "distance_km": distance_km,
                "co2e_kg": round(data.get("co2e", 0), 3),
                "co2e_unit": data.get("co2e_unit", "kg"),
                "co2e_source": "climatiq_api",
                "activity_id": activity_id
            }
        else:
            logger.error(f"CLIMATIQ API error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"CLIMATIQ API request exception: {e}")
        return None
    except Exception as e:
        logger.error(f"CLIMATIQ API unexpected error: {e}")
        return None


def _calculate_fallback_emission(transport_mode: str, distance_km: float) -> Dict[str, Any]:
    """
    Calculate carbon emissions using fallback emission factors.

    Args:
        transport_mode: Transport mode
        distance_km: Distance in kilometers

    Returns:
        Carbon emission data dict
    """
    emission_factor = FALLBACK_EMISSION_FACTORS.get(transport_mode, 0.1)
    co2e_kg = distance_km * emission_factor

    return {
        "transport_mode": transport_mode,
        "distance_km": distance_km,
        "co2e_kg": round(co2e_kg, 3),
        "co2e_unit": "kg",
        "co2e_source": "fallback_estimate",
        "emission_factor_per_km": emission_factor
    }


def add_carbon_to_transport_options(
    transport_options: Dict[str, Any],
    use_fallback: bool = False
) -> Dict[str, Any]:
    """
    Add carbon emission data to transport options.

    Args:
        transport_options: Dict of transport mode data
        use_fallback: Whether to use fallback calculation

    Returns:
        Updated transport options with carbon data
    """
    for mode, data in transport_options.items():
        if not data:
            continue

        distance_km = data.get("distance_km", 0)

        # Calculate carbon emission
        carbon_data = calculate_carbon_emission(mode, distance_km, use_fallback)

        if carbon_data:
            data["carbon_emission"] = carbon_data
            data["co2e_kg"] = carbon_data.get("co2e_kg", 0)
        else:
            data["carbon_emission"] = None
            data["co2e_kg"] = 0.0

    return transport_options


def compare_carbon_emissions(transport_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare carbon emissions across transport modes and rank them.

    Args:
        transport_options: Dict of transport mode data with carbon emissions

    Returns:
        Dict with comparison statistics
    """
    emissions = []

    for mode, data in transport_options.items():
        if data and "co2e_kg" in data:
            emissions.append({
                "mode": mode,
                "co2e_kg": data["co2e_kg"],
                "distance_km": data.get("distance_km", 0),
                "duration_minutes": data.get("duration_minutes", 0)
            })

    # Sort by emissions (ascending)
    emissions.sort(key=lambda x: x["co2e_kg"])

    # Calculate stats
    total_emissions = sum(e["co2e_kg"] for e in emissions)

    if emissions:
        lowest_emission = emissions[0]
        highest_emission = emissions[-1]

        return {
            "ranked_by_emissions": emissions,
            "lowest_emission_mode": lowest_emission["mode"],
            "lowest_emission_kg": lowest_emission["co2e_kg"],
            "highest_emission_mode": highest_emission["mode"],
            "highest_emission_kg": highest_emission["co2e_kg"],
            "total_options": len(emissions),
            "average_emission_kg": round(total_emissions / len(emissions), 3) if emissions else 0
        }
    else:
        return {
            "ranked_by_emissions": [],
            "lowest_emission_mode": None,
            "lowest_emission_kg": 0,
            "highest_emission_mode": None,
            "highest_emission_kg": 0,
            "total_options": 0,
            "average_emission_kg": 0
        }


def calculate_carbon_savings(
    chosen_mode: str,
    baseline_mode: str,
    transport_options: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Calculate carbon savings by choosing one mode over another.

    Args:
        chosen_mode: The mode actually chosen
        baseline_mode: The baseline mode to compare against (usually car/taxi)
        transport_options: Dict of transport mode data

    Returns:
        Dict with carbon savings data or None if modes not found
    """
    if chosen_mode not in transport_options or baseline_mode not in transport_options:
        return None

    chosen_data = transport_options[chosen_mode]
    baseline_data = transport_options[baseline_mode]

    if not chosen_data or not baseline_data:
        return None

    chosen_emission = chosen_data.get("co2e_kg", 0)
    baseline_emission = baseline_data.get("co2e_kg", 0)

    savings_kg = baseline_emission - chosen_emission
    savings_percent = (savings_kg / baseline_emission * 100) if baseline_emission > 0 else 0

    return {
        "chosen_mode": chosen_mode,
        "chosen_emission_kg": chosen_emission,
        "baseline_mode": baseline_mode,
        "baseline_emission_kg": baseline_emission,
        "savings_kg": round(savings_kg, 3),
        "savings_percent": round(savings_percent, 1),
        "is_lower_emission": savings_kg > 0
    }


def batch_calculate_carbon(
    route_results: list[Dict[str, Any]],
    use_fallback: bool = False
) -> list[Dict[str, Any]]:
    """
    Calculate carbon emissions for batch route results.

    Args:
        route_results: List of route result dicts from batch_process_routes
        use_fallback: Whether to use fallback calculation

    Returns:
        Updated route results with carbon data
    """
    for result in route_results:
        transport_options = result.get("transport_options", {})

        # Add carbon data to each transport option
        updated_options = add_carbon_to_transport_options(transport_options, use_fallback)
        result["transport_options"] = updated_options

        # Add carbon comparison
        result["carbon_comparison"] = compare_carbon_emissions(updated_options)

    return route_results


def get_low_carbon_recommendation(transport_options: Dict[str, Any]) -> Optional[str]:
    """
    Recommend the lowest carbon transport option that's still practical.

    Args:
        transport_options: Dict of filtered transport mode data with carbon emissions

    Returns:
        Recommended mode name or None
    """
    if not transport_options:
        return None

    # Get modes sorted by emissions
    modes_by_emission = sorted(
        transport_options.items(),
        key=lambda x: x[1].get("co2e_kg", float('inf'))
    )

    # Priority order for practical recommendations
    # (prefer active transport, then public, then taxi as last resort)
    priority_order = ["WALK", "TWO_WHEELER", "TRANSIT", "DRIVE"]

    # Find the lowest emission mode that's in our priority list
    for mode, data in modes_by_emission:
        if mode in priority_order and data:
            return mode

    # If no priority mode found, return lowest emission
    if modes_by_emission:
        return modes_by_emission[0][0]

    return None
