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
        from tools import (
            get_transport_options_concurrent,
            batch_process_routes,
            carbon_estimate
        )
        # from carbon_calculator import (
        #     add_carbon_to_transport_options,
        #     batch_calculate_carbon,
        #     compare_carbon_emissions
        # )

        self.known_actions = {
            "get_transport_options": get_transport_options_concurrent,
            "batch_process_routes": batch_process_routes,
            "carbon_estimate": carbon_estimate,
            # "calculate_carbon_emissions": add_carbon_to_transport_options,
            # "batch_calculate_carbon": batch_calculate_carbon,
            # "compare_carbon_emissions": compare_carbon_emissions
        }

    def __call__(self, message: str) -> Dict:
        """Execute reasoning and return transport data."""
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": str(result)})
        return result

    def execute(self) -> Dict:
        """Execute with function calling support, return transport data."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=self.messages,
            tools=self.available_tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Handle function calls if present
        if message.tool_calls:
            # Add assistant message with tool calls
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            all_tool_results = []

            # Execute tool calls
            for tool_call in message.tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                all_tool_results.append({
                    "tool": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                    "result": tool_result
                })

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, default=str)
                })

            # Return results from tools
            return {
                "reasoning": message.content,
                "tool_results": all_tool_results,
                "raw_routes": self._extract_routes_from_results(all_tool_results)
            }

        return {
            "reasoning": message.content,
            "tool_results": [],
            "raw_routes": []
        }

    def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute tool call and return raw results."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        logger.info(f"TRANSPORT AGENT: Executing {tool_name}")

        if tool_name in self.known_actions:
            return self.known_actions[tool_name](**tool_args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _extract_routes_from_results(self, tool_results: List[Dict]) -> List[Dict]:
        """Extract all route data from tool results."""
        all_routes = []
        for result in tool_results:
            if result["tool"] == "batch_process_routes":
                routes = result.get("result", [])
                if isinstance(routes, list):
                    all_routes.extend(routes)
            elif result["tool"] == "batch_calculate_carbon":
                # This returns the routes with carbon data added
                routes = result.get("result", [])
                if isinstance(routes, list):
                    all_routes = routes  # Replace with carbon-enriched routes
        return all_routes

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

    def _define_transport_tools(self) -> List[Dict]:
        """Define transport tools for LLM function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_transport_options",
                    "description": "Get transport options (walking, public transport, taxi, bicycle) between two locations with distance, duration, and cost",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {
                                "type": "object",
                                "properties": {
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"}
                                },
                                "required": ["latitude", "longitude"],
                                "description": "Origin location coordinates"
                            },
                            "destination": {
                                "type": "object",
                                "properties": {
                                    "latitude": {"type": "number"},
                                    "longitude": {"type": "number"}
                                },
                                "required": ["latitude", "longitude"],
                                "description": "Destination location coordinates"
                            }
                        },
                        "required": ["origin", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "batch_process_routes",
                    "description": "Process multiple route pairs efficiently to get transport options for all location pairs at once",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_pairs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "origin": {"type": "object"},
                                        "destination": {"type": "object"},
                                        "origin_name": {"type": "string"},
                                        "destination_name": {"type": "string"}
                                    }
                                },
                                "description": "List of location pairs to process"
                            },
                            "concurrent": {
                                "type": "boolean",
                                "description": "Whether to use concurrent processing (default: true)",
                                "default": True
                            }
                        },
                        "required": ["location_pairs"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "batch_calculate_carbon",
                    "description": "Calculate carbon emissions for all transport routes using hardcoded emission factors (driving: 0.21, taxi: 0.22, bus: 0.09, mrt: 0.035, cycle: 0, walking: 0.02 kg CO2/km)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "route_results": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "List of route results from batch_process_routes"
                            }
                        },
                        "required": ["route_results"]
                    }
                }
            }
        ]

    def process_places_data(self, places_data: Dict) -> Dict[str, Any]:
        """
        Process places data and extract places from nested structure.

        Args:
            places_data: Input data with retrieval.places_matrix.candidates structure

        Returns:
            Dict with places list and metadata
        """
        # Extract places from nested structure
        retrieval = places_data.get("retrieval", {})
        places_matrix = retrieval.get("places_matrix", {})
        # Support both "candidates" (new) and "nodes" (old) for backward compatibility
        formatted_places = places_matrix.get("candidates", places_matrix.get("nodes", []))

        logger.info(f"Loaded {len(formatted_places)} places from input")

        return {
            "formatted_places": formatted_places,
            "total_places": len(formatted_places),
            "metadata": {
                "attractions_count": retrieval.get("attractions_count", 0),
                "food_count": retrieval.get("food_count", 0),
                "places_found": retrieval.get("places_found", 0)
            }
        }

    def identify_location_pairs(
        self,
        places: List[Dict],
        accommodation: Optional[Dict[str, float]] = None
    ) -> list:
        """
        Identify all unique location pairs that need transport calculations.
        Creates one-way connections: A→B, A→C, B→C (but not B→A, C→A, C→B).

        Args:
            places: List of all places
            accommodation: Optional accommodation location dict

        Returns:
            List of tuples (origin_coords, dest_coords, origin_name, dest_name)
        """
        pairs = []

        # Add accommodation to all places if provided
        if accommodation:
            acc_location = {
                "latitude": accommodation.get("lat"),
                "longitude": accommodation.get("lon", accommodation.get("lng"))
            }

            for destination_place in places:
                destination_location = destination_place.get("geo", {})
                pairs.append((
                    acc_location,
                    destination_location,
                    "Accommodation",
                    destination_place.get("name", "Unknown")
                ))

        # Add all place-to-place pairs (one-way only)
        # A→B, A→C, A→D, B→C, B→D, C→D
        for i, origin_place in enumerate(places):
            for destination_place in places[i+1:]:
                pairs.append((
                    origin_place.get("geo", {}),
                    destination_place.get("geo", {}),
                    origin_place.get("name", "Unknown"),
                    destination_place.get("name", "Unknown")
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
            location_pairs: List of 4-element tuples (origin_coords, dest_coords, origin_name, dest_name)
            concurrent: Whether to use concurrent processing

        Returns:
            List of route results with transport options and carbon data
        """
        from tools import batch_process_routes
        from carbon_calculator import batch_calculate_carbon

        logger.info(f"Calculating routes for {len(location_pairs)} location pairs...")

        # Get route data for all pairs
        route_results = batch_process_routes(location_pairs, concurrent=concurrent)

        logger.info(f"Adding carbon emission data...")

        # Add carbon emissions to all routes using hardcoded emission factors
        route_results = batch_calculate_carbon(route_results)

        logger.info(f"Completed route calculations")

        return route_results

    def format_output(self, route_results: list, metadata: Dict, places_data: Dict) -> Dict:
        """
        Format final output with all transport and carbon data.

        Args:
            route_results: List of route results with transport options
            metadata: Metadata about the input data (not used, kept for compatibility)
            places_data: Original places data with requirements and retrieval structure

        Returns:
            Formatted output dict preserving input structure with added connection_matrix
        """
        # Extract formatted places from nested structure
        retrieval = places_data.get("retrieval", {})
        places_matrix = retrieval.get("places_matrix", {})
        # Support both "candidates" (new) and "nodes" (old) for backward compatibility
        formatted_places = places_matrix.get("candidates", places_matrix.get("nodes", []))

        # Create mapping from place name to place_id
        name_to_id = {}
        for place in formatted_places:
            name_to_id[place.get("name")] = place.get("place_id")

        # Build connection_matrix structure
        connections = []
        connection_id = 1

        for route_result in route_results:
            origin_name = route_result.get("origin")
            destination_name = route_result.get("destination")
            transport_options = route_result.get("transport_options", {})

            # Get place IDs
            if origin_name == "Accommodation":
                from_place_id = "accommodation"
                to_place_id = name_to_id.get(destination_name)
            else:
                from_place_id = name_to_id.get(origin_name)
                to_place_id = name_to_id.get(destination_name)

            # Skip if we can't find IDs (shouldn't happen)
            if not to_place_id or (origin_name != "Accommodation" and not from_place_id):
                logger.warning(f"Could not find place_id for {origin_name} -> {destination_name}")
                continue

            # Format transport modes
            transport_modes = []
            for mode, mode_data in transport_options.items():
                # Map API mode names to user-friendly names
                mode_map = {
                    "WALK": "walking",
                    "TRANSIT": "public_transport",
                    "DRIVE": "taxi",
                    "CYCLING": "cycle"  # Renamed from cycling to cycle
                }
                friendly_mode = mode_map.get(mode, mode.lower())

                # For TRANSIT mode, determine if it's mrt, bus, or public_transport
                if mode == "TRANSIT" and "transit_summary" in mode_data:
                    transit_summary = mode_data.get("transit_summary", "").lower()
                    # Check if it contains both MRT and Bus
                    has_mrt = "mrt" in transit_summary or "metro" in transit_summary or "train" in transit_summary
                    has_bus = "bus" in transit_summary

                    if has_mrt and has_bus:
                        friendly_mode = "public_transport"
                    elif has_mrt:
                        friendly_mode = "mrt"
                    elif has_bus:
                        friendly_mode = "bus"
                    else:
                        friendly_mode = "public_transport"

                distance_km = mode_data.get("distance_km", 0)
                duration_minutes = mode_data.get("duration_minutes", 0)
                cost_sgd = mode_data.get("estimated_cost_sgd", 0.0)

                # Calculate carbon emissions using carbon_estimate
                from tools import carbon_estimate
                carbon_kg = carbon_estimate(friendly_mode, distance_km)

                # Create route summary
                route_summary = f"{distance_km} km, {duration_minutes:.0f} mins via {friendly_mode}"

                transport_mode_entry = {
                    "mode": friendly_mode,
                    "distance_km": distance_km,
                    "duration_minutes": round(duration_minutes, 1),
                    "cost_sgd": round(cost_sgd, 2),
                    "carbon_kg": round(carbon_kg, 3),
                    "route_summary": route_summary
                }

                # Add transit summary if available (for all transit modes)
                if mode == "TRANSIT" and "transit_summary" in mode_data:
                    transport_mode_entry["transit_summary"] = mode_data["transit_summary"]
                    transport_mode_entry["num_transfers"] = mode_data.get("num_transfers", 0)

                # Add note for cycle (custom category)
                if mode == "CYCLING":
                    transport_mode_entry["note"] = mode_data.get("note", "")

                transport_modes.append(transport_mode_entry)

            # Add connection entry
            connection = {
                "connection_id": connection_id,
                "from_place_id": from_place_id,
                "to_place_id": to_place_id,
                "from_place_name": origin_name,
                "to_place_name": destination_name,
                "transport_modes": transport_modes
            }
            connections.append(connection)
            connection_id += 1

        # Create output structure preserving input structure
        # Deep copy retrieval object and update structure
        import copy
        retrieval = copy.deepcopy(places_data.get("retrieval", {}))

        # Move candidates from places_matrix to same level as conditions
        if "places_matrix" in retrieval:
            places_matrix = retrieval.pop("places_matrix")
            # Support both "candidates" (new) and "nodes" (old) for backward compatibility
            candidates = places_matrix.get("candidates", places_matrix.get("nodes", []))
            retrieval["candidates"] = candidates

        # Add connections at same level as conditions (not nested in connection_matrix)
        retrieval["total_unique_connections"] = len(connections)
        retrieval["connections"] = connections

        output = {
            "requirements": places_data.get("requirements", {}),
            "retrieval": retrieval
        }

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
    from tools import clear_raw_responses, dump_raw_responses

    start_time = time.time()

    # Clear any previous raw responses
    clear_raw_responses()
    logger.info("Cleared previous raw Google Maps responses")

    # Load input data
    logger.info(f"Loading input file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            places_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    # Extract accommodation location from input file if not provided
    if accommodation_location is None:
        optional = places_data.get("requirements", {}).get("optional", {})
        input_accommodation = optional.get("accommodation_location", {})
        if input_accommodation:
            accommodation_location = {
                "lat": input_accommodation.get("lat"),
                "lon": input_accommodation.get("lon", input_accommodation.get("lng"))
            }
            logger.info(f"Using accommodation location from input file: {accommodation_location}")

    # Initialize agent
    logger.info("Initializing Transport Sustainability Agent")
    agent = TransportSustainabilityAgent()

    # Process places data
    logger.info("Processing places data...")
    processed_data = agent.process_places_data(places_data)

    # Identify location pairs
    logger.info("Identifying location pairs...")
    location_pairs = agent.identify_location_pairs(
        processed_data["formatted_places"],
        accommodation_location
    )

    # Calculate all routes
    logger.info("Calculating routes and carbon emissions...")
    route_results = agent.calculate_all_routes(location_pairs, concurrent=True)

    # Format output
    logger.info("Formatting output...")
    output = agent.format_output(route_results, processed_data["metadata"], places_data)

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

    # Dump raw Google Maps responses
    raw_output_file = output_file.replace(".json", "_raw_responses.json")
    dump_raw_responses(raw_output_file)

    # Print summary
    print(f"\n{'='*80}")
    print(f"TRANSPORT PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total location pairs processed: {len(route_results)}")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Output saved to: {output_file}")
    print(f"Raw Google Maps responses: {raw_output_file}")
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