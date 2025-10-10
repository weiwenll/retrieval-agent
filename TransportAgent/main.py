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
        
        # Tool definitions for Google Places API
        self.available_tools = self._define_places_tools()
        
        # Known actions mapping
        self.known_actions = {
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