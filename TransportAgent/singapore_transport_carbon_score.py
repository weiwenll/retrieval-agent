"""
Singapore Transport Carbon Calculator
==========================================================

Module for carbon emission calculations for Singapore transport modes.
Conservative method for sustainability claims.

Version: 1.0.0
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TransportLeg:
    """Single leg of a journey"""
    mode: str
    distance_km: float
    passengers: int = 1


@dataclass
class CarbonResult:
    """Carbon emission calculation result"""
    # Core metrics
    total_co2e_kg: float
    co2e_per_person_kg: float
    co2e_per_km_g: float
    grade: str
    
    # Optional details
    vs_mrt_multiplier: Optional[float] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self):
        return asdict(self)


class SingaporeTransportCarbon:
    """
    Singapore transport carbon calculator

    Usage:
        calculator = SingaporeTransportCarbon()
        result = calculator.calculate(mode="car_petrol", distance_km=10, passengers=1, is_taxi=True)
        print(f"CO2e: {result.total_co2e_kg} kg, Grade: {result.grade}")
    """
    
    # Emission factors (g CO2e per passenger-km)
    FACTORS = {
        "mrt": 18,
        "lrt": 18,
        "bus": 75,
        "car_petrol": 150,
        "car_diesel": 133,
        "car_electric": 63,
        "bicycle": 0,
        "walk": 0,
    }
    
    # Grades (g CO2e per km)
    GRADES = [
        (0, 10, "A+"),
        (10, 25, "A"),
        (25, 60, "B"),
        (60, 100, "C"),
        (100, 150, "D"),
        (150, 9999, "E"),
    ]
    
    def calculate(
        self,
        mode: str,
        distance_km: float,
        passengers: int = 1,
        is_taxi: bool = False,
        traffic: str = "normal"
    ) -> CarbonResult:
        """
        Calculate carbon emissions
        
        Args:
            mode: mrt, lrt, bus, car_petrol, car_diesel, car_electric, bicycle, walk
            distance_km: Distance in kilometers
            passengers: Number of people traveling
            is_taxi: True for taxi/Grab (adds 30% for empty return trips)
            traffic: light, normal, heavy, peak (affects cars)
        
        Returns:
            CarbonResult with emissions and grade
        """
        # Validate
        if mode not in self.FACTORS:
            raise ValueError(f"Invalid mode. Choose from: {list(self.FACTORS.keys())}")
        if distance_km <= 0:
            raise ValueError("Distance must be positive")
        if passengers < 1:
            raise ValueError("Passengers must be >= 1")
        
        # Get base factor
        factor = self.FACTORS[mode]
        
        # Apply taxi return trip multiplier
        if is_taxi and mode.startswith("car_"):
            factor *= 1.3
        
        # Apply traffic multiplier for cars
        if mode.startswith("car_"):
            traffic_mult = {"light": 1.0, "normal": 1.1, "heavy": 1.2, "peak": 1.3}
            factor *= traffic_mult.get(traffic, 1.1)
        
        # Calculate emissions
        if mode in ["mrt", "lrt", "bus"]:
            co2e_per_person_g = factor * distance_km
            total_co2e_g = co2e_per_person_g * passengers
        else:
            total_co2e_g = factor * distance_km * passengers
            co2e_per_person_g = total_co2e_g / passengers
        
        # Per km
        co2e_per_km_g = co2e_per_person_g / distance_km
        
        # Grade
        grade = self._get_grade(co2e_per_km_g)
        
        # vs MRT
        mrt_factor = self.FACTORS["mrt"]
        vs_mrt = round(co2e_per_km_g / mrt_factor, 2) if mrt_factor > 0 else None
        
        # Warnings
        warnings = []
        if is_taxi:
            warnings.append("Includes 30% for empty taxi trips")
        if mode.startswith("car_") and passengers == 1:
            warnings.append("Single occupancy - carpooling reduces emissions")
        
        return CarbonResult(
            total_co2e_kg=round(total_co2e_g / 1000, 3),
            co2e_per_person_kg=round(co2e_per_person_g / 1000, 3),
            co2e_per_km_g=round(co2e_per_km_g, 1),
            grade=grade,
            vs_mrt_multiplier=vs_mrt,
            warnings=warnings if warnings else None
        )
    
    def calculate_multimodal(
        self,
        legs: List[TransportLeg],
        is_taxi: Optional[List[bool]] = None
    ) -> Dict:
        """
        Calculate multi-modal journey
        
        Args:
            legs: List of TransportLeg objects
            is_taxi: Optional list indicating which car legs are taxis
        
        Returns:
            Dict with total and breakdown
        """
        if is_taxi is None:
            is_taxi = [False] * len(legs)
        
        results = []
        total_distance = 0
        total_co2e = 0
        
        for i, leg in enumerate(legs):
            result = self.calculate(
                mode=leg.mode,
                distance_km=leg.distance_km,
                passengers=leg.passengers,
                is_taxi=is_taxi[i] if i < len(is_taxi) else False
            )
            results.append(result.to_dict())
            total_distance += leg.distance_km
            total_co2e += result.total_co2e_kg
        
        avg_per_km = (total_co2e * 1000) / total_distance if total_distance > 0 else 0
        
        return {
            "total_distance_km": round(total_distance, 2),
            "total_co2e_kg": round(total_co2e, 3),
            "avg_co2e_per_km_g": round(avg_per_km, 1),
            "grade": self._get_grade(avg_per_km),
            "legs": results
        }
    
    def _get_grade(self, co2e_per_km_g: float) -> str:
        """Get sustainability grade"""
        for min_val, max_val, grade in self.GRADES:
            if min_val <= co2e_per_km_g < max_val:
                return grade
        return "E"


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def carbon_estimate(mode: str, distance_km: float) -> float:
    """
    Calculate carbon emissions for a given transport mode and distance.

    Args:
        mode: Transport mode (mrt, bus, taxi, walking, driving, cycle, etc.)
        distance_km: Distance in kilometers

    Returns:
        Carbon emissions in kg CO2
    """
    # Map mode names to singapore_transport_carbon_score modes
    mode_mapping = {
        "walking": "walk",
        "walk": "walk",
        "cycle": "bicycle",
        "bicycle": "bicycle",
        "cycling": "bicycle",
        "taxi": "car_petrol",  # Default taxi to petrol
        "driving": "car_petrol",
        "drive": "car_petrol",
        "ride": "car_petrol",  # Ride-hailing (Grab/taxi)
        "car": "car_petrol",
        "mrt": "mrt",
        "lrt": "lrt",
        "bus": "bus",
        "transit": "mrt",  # Default transit to MRT
        "public_transport": "bus",  # Default to bus
    }

    standard_mode = mode_mapping.get(mode.lower(), "car_petrol")

    # Use the calculator
    calculator = SingaporeTransportCarbon()

    # Special handling for taxi/ride-hailing (add 30% multiplier for empty return trips)
    is_taxi = mode.lower() in ["taxi", "grab", "uber", "ride"]

    try:
        result = calculator.calculate(
            mode=standard_mode,
            distance_km=distance_km,
            passengers=1,
            is_taxi=is_taxi
        )
        return result.total_co2e_kg
    except Exception as e:
        # Fallback to simple calculation if error
        factor_kg = calculator.FACTORS.get(standard_mode, 150) / 1000  # Convert g to kg
        return distance_km * factor_kg


def calculate_carbon_emission(transport_mode: str, distance_km: float) -> Dict:
    """
    Calculate carbon emissions with detailed output.

    Args:
        transport_mode: Transport mode
        distance_km: Distance in kilometers

    Returns:
        Dict with carbon emission details including mode, distance, co2e, unit, and source
    """
    carbon_kg = carbon_estimate(transport_mode, distance_km)

    return {
        "transport_mode": transport_mode.lower(),
        "distance_km": round(distance_km, 2),
        "co2e_kg": round(carbon_kg, 3),
        "co2e_unit": "kg",
        "co2e_source": "singapore_lta_nea_2024"
    }


# Quick test
if __name__ == "__main__":
    api = SingaporeTransportCarbon()
    
    # Example 1: Grab ride
    result = api.calculate(
        mode="car_petrol",
        distance_km=10,
        passengers=1,
        is_taxi=True
    )
    print(f"Grab (10km): {result.total_co2e_kg}kg CO2e, Grade {result.grade}")
    
    # Example 2: MRT
    result = api.calculate(mode="mrt", distance_km=15, passengers=1)
    print(f"MRT (15km): {result.total_co2e_kg}kg CO2e, Grade {result.grade}")
    
    # Example 3: Multi-modal
    journey = [
        TransportLeg(mode="walk", distance_km=0.5),
        TransportLeg(mode="mrt", distance_km=12),
        TransportLeg(mode="walk", distance_km=0.3)
    ]
    result = api.calculate_multimodal(journey)
    print(f"Multi-modal: {result['total_co2e_kg']}kg CO2e, Grade {result['grade']}")
