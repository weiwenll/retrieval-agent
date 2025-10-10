# # Singapore Transport Agent
# Agentic system for discovering transport modes and carbon scoring for getting from accommodation to attractions and between attractions in Singapore.

# Install requirements if needed
# !pip install -r requirements.txt

# Imports
import os
import sys
import json
import logging
import math
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname('.'), 'src'))

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_reasoning.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check API keys
print("API Keys Status:")
print(f"GOOGLE_MAPS_API_KEY: {'Set' if os.getenv('GOOGLE_MAPS_API_KEY') else 'Missing'}")
print(f"CLIMATIQ_API_KEY: {'Set' if os.getenv('CLIMATIQ_API_KEY') else 'Missing'}")
print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Missing'}")

# ## Input Data Loading
# Load and validate input file with trip requirements.

def load_input_file(file_path: str) -> dict:
    """Load and validate input file."""
    try:
        with open(file_path, 'r') as f:
            input_data = json.load(f)
        
        # Basic validation
        required_fields = ["trip_dates", "duration_days", "budget", "pace"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Check if optional section exists, and validate accommodation_location if it does
        if "optional" in input_data and "accommodation_location" not in input_data["optional"]:
            raise ValueError("Missing required field: optional.accommodation_location")
        
        return input_data
    
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Test with different input files
print("Available input files:")
try:
    for file in os.listdir('../inputs'):
        if file.endswith('.json'):
            print(f"  ../inputs/{file}")
except FileNotFoundError:
    print("No inputs directory found")
    
class TransportSustainabilityAgent:
    def __init__(self, system_prompt=""):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Agent configuration
        self.model_name = "gpt-4o"
        self.temperature = 0.3
        
        # Message history
        self.messages = []
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        
        # Tool definitions
        self.available_tools = self._define_transport_tools()

        # Known actions mapping (import functions from other modules)
        from transport_tools import (
            get_transport_options_concurrent,
            batch_process_routes,
            filter_transport_options
        )
        from carbon_calculator import (
            add_carbon_to_transport_options,
            batch_calculate_carbon,
            compare_carbon_emissions
        )

        self.known_actions = {
            "get_transport_options": get_transport_options_concurrent,
            "calculate_carbon_emissions": add_carbon_to_transport_options,
            "batch_process_routes": batch_process_routes,
            "filter_transport_options": filter_transport_options,
            "batch_calculate_carbon": batch_calculate_carbon,
            "compare_carbon_emissions": compare_carbon_emissions
        }
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt following thought-action-observation pattern."""
        return """
            You are a transportation routing specialist. You run in a loop of Thought, Action, Observation.
            At the end of the loop you output raw transport data results with carbon emissions.

            Use Thought to describe your reasoning about transport routing and optimization.
            Use Action to run one of the actions available to you.
            Observation will be the result of running those actions.

            INPUT PROCESSING:
            1. Group all attractions/places by their geo_cluster_id
            2. Include accommodation details in each geo_cluster_group
            3. Identify all unique location pairs that need transport calculations

            Your available actions are:
            1. get_transport_options: Get all transport modes between two locations
            2. calculate_carbon_emissions: Calculate carbon score for each transport mode
            3. batch_process_routes: Process multiple route pairs efficiently (async when possible)

            TRANSPORT MODE EVALUATION:
            For each location pair, retrieve and evaluate ALL transport modes:
            - Walking
            - Public Transit (MRT/Bus) 
            - Taxi/Ride-hailing
            - Driving (if applicable)

            TRANSPORT FILTERING RULES:
            Apply intelligent filtering based on practicality thresholds:

            Walking:
            - Exclude if distance > 2.0 km
            - Exclude if duration > 25 minutes
            - Consider weather/climate factors if available

            Public Transit (MRT/Bus):
            - Exclude if requires > 3 transfers
            - Exclude if duration > 60 minutes
            - Exclude if walking portion > 1.5 km
            - Flag if service hours limited (early morning/late night)

            Taxi/Ride-hailing:
            - Always include as fallback option
            - Flag if cost > 30 SGD as "expensive"
            - Consider surge pricing times if data available

            CARBON EMISSIONS CALCULATION:
            For each valid transport mode:
            1. Use latitude/longitude of origin and destination
            2. Call CLIMATIQ_API with:
            - Transport mode type
            - Distance traveled
            - Vehicle type (if applicable)
            3. Include carbon score in final output
        """.strip()

    def _define_transport_tools(self) -> list:
        """Define available transport tools for the agent."""
        return []  # Tool definitions not used in this implementation

    def process_places_data(self, places_data: Dict) -> Dict[str, Any]:
        """
        Process places data and group by geo_cluster_id.

        Args:
            places_data: Input data with formatted_places list

        Returns:
            Dict with grouped places and metadata
        """
        formatted_places = places_data.get("formatted_places", [])

        # Group places by geo_cluster_id
        clusters = {}
        for place in formatted_places:
            cluster_id = place.get("geo_cluster_id", "unknown")
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(place)

        logger.info(f"Grouped {len(formatted_places)} places into {len(clusters)} clusters")

        return {
            "clusters": clusters,
            "total_places": len(formatted_places),
            "cluster_count": len(clusters),
            "metadata": {
                "attractions_count": places_data.get("attractions_count", 0),
                "food_count": places_data.get("food_count", 0),
                "places_found": places_data.get("places_found", 0)
            }
        }

    def identify_location_pairs(
        self,
        clusters: Dict[str, list],
        accommodation: Optional[Dict[str, float]] = None
    ) -> list:
        """
        Identify all unique location pairs that need transport calculations.

        Args:
            clusters: Dict of places grouped by cluster_id
            accommodation: Optional accommodation location dict

        Returns:
            List of tuples (origin, destination, origin_name, dest_name)
        """
        pairs = []

        # Get all places as a flat list
        all_places = []
        for cluster_places in clusters.values():
            all_places.extend(cluster_places)

        # Add accommodation to pairs if provided
        if accommodation:
            acc_location = {
                "latitude": accommodation.get("lat"),
                "longitude": accommodation.get("lon", accommodation.get("lng"))
            }

            for place in all_places:
                place_location = place.get("geo", {})
                pairs.append((
                    acc_location,
                    place_location,
                    "Accommodation",
                    place.get("name", "Unknown")
                ))

        # Add inter-attraction pairs (within same cluster to keep it manageable)
        for cluster_id, cluster_places in clusters.items():
            for i, place1 in enumerate(cluster_places):
                for place2 in cluster_places[i+1:]:
                    pairs.append((
                        place1.get("geo", {}),
                        place2.get("geo", {}),
                        place1.get("name", "Unknown"),
                        place2.get("name", "Unknown")
                    ))

        logger.info(f"Identified {len(pairs)} location pairs for transport calculation")

        return pairs

    def calculate_all_routes(
        self,
        location_pairs: list,
        concurrent: bool = True
    ) -> list:
        """
        Calculate routes and carbon emissions for all location pairs.

        Args:
            location_pairs: List of location pair tuples
            concurrent: Whether to use concurrent processing

        Returns:
            List of route results with transport options and carbon data
        """
        from transport_tools import batch_process_routes
        from carbon_calculator import batch_calculate_carbon

        logger.info(f"Calculating routes for {len(location_pairs)} location pairs...")

        # Get route data for all pairs
        route_results = batch_process_routes(location_pairs, concurrent=concurrent)

        logger.info(f"Adding carbon emission data...")

        # Add carbon emissions to all routes
        route_results = batch_calculate_carbon(route_results, use_fallback=False)

        logger.info(f"Completed route calculations")

        return route_results

    def format_output(self, route_results: list, metadata: Dict) -> Dict:
        """
        Format final output with all transport and carbon data.

        Args:
            route_results: List of route results with transport options
            metadata: Metadata about the input data

        Returns:
            Formatted output dict
        """
        output = {
            "metadata": metadata,
            "total_route_pairs": len(route_results),
            "route_data": []
        }

        for result in route_results:
            transport_options = result.get("transport_options", {})
            carbon_comparison = result.get("carbon_comparison", {})

            route_entry = {
                "origin": result.get("origin"),
                "destination": result.get("destination"),
                "origin_coords": result.get("origin_coords"),
                "destination_coords": result.get("destination_coords"),
                "available_transport_modes": list(transport_options.keys()),
                "transport_details": transport_options,
                "carbon_comparison": carbon_comparison
            }

            output["route_data"].append(route_entry)

        return output


def process_transport_data(input_file: str, output_file: str, accommodation_location: Optional[Dict] = None):
    """
    Main function to process places data and calculate transport options.

    Args:
        input_file: Path to input JSON file with places data
        output_file: Path to output JSON file
        accommodation_location: Optional accommodation location dict
    """
    import json
    import time

    start_time = time.time()

    # Load input data
    logger.info(f"Loading input file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            places_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    # Initialize agent
    logger.info("Initializing Transport Sustainability Agent")
    agent = TransportSustainabilityAgent()

    # Process places data
    logger.info("Processing places data...")
    processed_data = agent.process_places_data(places_data)

    # Identify location pairs
    logger.info("Identifying location pairs...")
    location_pairs = agent.identify_location_pairs(
        processed_data["clusters"],
        accommodation_location
    )

    # Calculate all routes
    logger.info("Calculating routes and carbon emissions...")
    route_results = agent.calculate_all_routes(location_pairs, concurrent=True)

    # Format output
    logger.info("Formatting output...")
    output = agent.format_output(route_results, processed_data["metadata"])

    # Add timing info
    elapsed_time = time.time() - start_time
    output["processing_time_seconds"] = round(elapsed_time, 2)

    # Save output
    logger.info(f"Saving output to: {output_file}")
    try:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Successfully saved output")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"TRANSPORT PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total location pairs processed: {len(route_results)}")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}\n")


# Main execution
if __name__ == "__main__":
    import sys

    # Check if input file provided
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file> [output_file] [--accommodation lat,lon]")
        print("\nExample:")
        print("  python main.py inputs/attractions_12_food_4.json output.json --accommodation 1.3294,103.8021")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "transport_output.json"

    # Parse accommodation location if provided
    accommodation = None
    if "--accommodation" in sys.argv:
        acc_idx = sys.argv.index("--accommodation")
        if acc_idx + 1 < len(sys.argv):
            acc_coords = sys.argv[acc_idx + 1].split(",")
            if len(acc_coords) == 2:
                try:
                    accommodation = {
                        "lat": float(acc_coords[0]),
                        "lon": float(acc_coords[1])
                    }
                    print(f"Using accommodation location: {accommodation}")
                except ValueError:
                    print("Warning: Invalid accommodation coordinates, ignoring")

    # Process transport data
    process_transport_data(input_file, output_file, accommodation)