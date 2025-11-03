"""
Singapore Place-Based Carbon Calculator
==========================================================

Module for onsite carbon emission estimates for different place types in Singapore.
Carbon estimates represent typical emissions per visit (kg CO2e).

Based on Google Places API primary types.

Version: 1.0.0
"""

from typing import Dict, Optional
from dataclasses import dataclass
from singapore_onsite_carbon_config import PLACE_CARBON_FACTORS


@dataclass
class PlaceCarbonResult:
    """Carbon emission result for a place visit"""
    place_type: str
    co2e_per_visit_kg: float
    co2e_total_kg: float  # Total for all people
    num_people: int
    low_carbon_score: float
    confidence: str  # "high", "medium", "low", "estimated"
    notes: Optional[str] = None


class SingaporePlaceCarbon:
    """
    Singapore place-based carbon calculator

    Usage:
        calculator = SingaporePlaceCarbon()
        result = calculator.calculate(primary_type="museum", num_people=2)
        print(f"CO2e per visit: {result.co2e_per_visit_kg} kg")
        print(f"Total CO2e: {result.co2e_total_kg} kg")
    """

    def calculate(
        self,
        primary_type: str,
        num_people: int = 1
    ) -> PlaceCarbonResult:
        """
        Calculate carbon emissions for a place visit.

        Args:
            primary_type: Google Places API primaryType
            num_people: Number of people visiting (adults + children)

        Returns:
            PlaceCarbonResult with emissions estimate
        """
        # Get carbon data
        place_data = PLACE_CARBON_FACTORS.get(primary_type)

        if place_data and place_data["co2e_kg"] is not None:
            co2e_per_visit = place_data["co2e_kg"]
            co2e_total = co2e_per_visit * num_people
            low_carbon_score = place_data.get("low_carbon_score")

            return PlaceCarbonResult(
                place_type=primary_type,
                co2e_per_visit_kg=co2e_per_visit,
                co2e_total_kg=co2e_total,
                num_people=num_people,
                low_carbon_score=low_carbon_score,
                confidence="medium",
                notes=place_data.get("notes")
            )

        # Fallback to zero if no data
        return PlaceCarbonResult(
            place_type=primary_type,
            co2e_per_visit_kg=0.0,
            co2e_total_kg=0.0,
            num_people=num_people,
            low_carbon_score=50.0,
            confidence="low",
            notes="No data available for this place type"
        )

    def get_low_carbon_score(
        self,
        primary_type: str,
        num_people: int = 1
    ) -> float:
        """
        Get low carbon score for a place.

        Args:
            primary_type: Google Places API primaryType
            num_people: Number of people visiting

        Returns:
            Low carbon score (0-100)
        """
        result = self.calculate(primary_type, num_people)
        return result.low_carbon_score


# ===================================================================
# PUBLIC API
# ===================================================================

def get_place_carbon_details(primary_type: str, num_people: int = 1) -> Dict:
    """
    Get detailed carbon information for a place type.

    Args:
        primary_type: Google Places API primaryType
        num_people: Number of people visiting

    Returns:
        Dict with carbon details including emissions, score, and notes
    """
    calculator = SingaporePlaceCarbon()
    result = calculator.calculate(primary_type, num_people)
    return {
        "place_type": result.place_type,
        "co2e_per_visit_kg": result.co2e_per_visit_kg,
        "co2e_total_kg": result.co2e_total_kg,
        "num_people": result.num_people,
        "low_carbon_score": result.low_carbon_score,
        "confidence": result.confidence,
        "notes": result.notes
    }


# Quick test
if __name__ == "__main__":
    calculator = SingaporePlaceCarbon()

    # Example 1: Natural park (low carbon)
    result = calculator.calculate("park", num_people=2)
    print(f"Park (2 people): {result.co2e_total_kg}kg CO2e total, Score: {result.low_carbon_score}")

    # Example 2: Museum
    result = calculator.calculate("museum", num_people=3)
    print(f"Museum (3 people): {result.co2e_total_kg}kg CO2e total, Score: {result.low_carbon_score}")

    # Example 3: Restaurant
    result = calculator.calculate("restaurant", num_people=5)
    print(f"Restaurant (5 people): {result.co2e_total_kg}kg CO2e total, Score: {result.low_carbon_score}")
