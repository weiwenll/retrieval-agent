# # Singapore Attraction Discovery Agent
# Agentic system for discovering Singapore attractions using tool-based reasoning, LLM decision-making, and iterative refinement to meet user requirements.

# Install requirements if needed
# !pip install -r requirements.txt

# Imports
import os
import sys
import json
import logging
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
print(f"GOOGLE_MAPS_API_KEY: {'✓ Set' if os.getenv('GOOGLE_MAPS_API_KEY') else '✗ Missing'}")
print(f"CLIMATIQ_API_KEY: {'✓ Set' if os.getenv('CLIMATIQ_API_KEY') else '✗ Missing'}")
print(f"OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
print(f"ANTHROPIC_API_KEY: {'✓ Set' if os.getenv('ANTHROPIC_API_KEY') else '✗ Missing'}")

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
for file in os.listdir('inputs'):
    if file.endswith('.json'):
        print(f"  inputs/{file}")
        
# ## Places Research Agent - Reasoning Implementation
# - Handles pure reasoning and analytical thinking
# - Executes API calls through tools (search_places, search_multiple_keywords, get_place_details)
# - Returns raw data objects from API calls without any formatting
# - Has the calculate_required_places method

from tools import search_places, search_multiple_keywords, get_place_details

class PlacesResearchReasoningAgent:
    """
    Specialized agent for reasoning about location-based attraction research.
    Returns raw data from Google Places API without formatting.
    """

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
            "search_places": search_places,
            "search_multiple_keywords": search_multiple_keywords,
            "get_place_details": get_place_details
        }

        # Agent state for tracking requirements
        self.required_places = 0
        self.current_results_count = 0
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt following thought-action-observation pattern."""
        return """
            You are a places and attractions research specialist. You run in a loop of Thought, Action, Observation.
            At the end of the loop you output raw data results.
            
            Use Thought to describe your reasoning about the places and attractions research.
            Use Action to run one of the actions available to you.
            Observation will be the result of running those actions.
            
            INTEREST MAPPING RULES:
            Before processing any interests, map them to the following standardized categories:
            - tourist_attraction
            - food
            - cafe
            - bar
            - bakery
            - park
            - museum
            - shopping_mall
            - lodging
            
            Mapping examples:
            - "parks" to park
            - "museums" to museum
            - "family" to tourist_attraction
            - "educational" to museum or tourist_attraction
            - "gardens" to park
            - "nature" to park
            - "culture" to museum or 'cultural food'
            - "dining" to food
            - "coffee" to cafe
            - "shopping" to shopping_mall
            - "accommodation" to lodging
            
            Special handling for food-related interests:
            If an interest cannot be classified into the above categories but refers to food options 
            (like "vegetarian", "halal", "vegan", "local cuisine"), rewrite it as "[interest] food".
            Examples:
            - "vegetarian" → "vegetarian food"
            - "halal" → "halal food"
            - "local cuisine" → "local food"
            
            Your available actions are:
            1. search_places when there is only ONE interest/keyword to search for.
            2. search_multiple_keywords when there are MULTIPLE interests/keywords to search for.
            3. get_place_details to get comprehensive details about all places.
            
            The 'interests' input from user maps to:
            - 'keyword' parameter for search_places (single string)
            - 'keywords' parameter for search_multiple_keywords (array of strings)

            Finds attractions near accomodation location and find at least one food places per day.
            
            Set ratings filter to a 3.5 for shopping malls.
            Set ratings filter to a 4.2 for food places.
            Set ratings filter to a 4.0 for all other place types.
            
            Return raw data objects from API calls. Do not format the output.
            Focus on comprehensive data collection based on user interests and accommodation location.
            Always consider practical factors like location, ratings, accessibility needs, and user preferences.
            Calculate required places based on pace and duration, and expand search if insufficient results are found.
            """.strip()

    def _define_places_tools(self) -> List[Dict]:
        """Define Google Places API tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_places",
                    "description": "Search for places with a SINGLE keyword/interest based on user interests and accommodation location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number", "description": "Latitude of accommodation"},
                                    "lng": {"type": "number", "description": "Longitude of accommodation"},
                                    # "neighborhood": {"type": "string", "description": "Neighborhood name (optional)"}
                                },
                                "required": ["lat", "lng"],
                                "description": "Accommodation location with coordinates"
                            },
                            "keyword": {
                                "type": "string",
                                "description": "A single user interest like 'tourist_attraction'"
                            },
                            "radius": {
                                "type": "integer",
                                "description": "Search radius in meters (default: 5000, increase by 2000 if not enough results)",
                                "default": 5000
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum number of pages to retrieve (default: 3), Each page has up to 20 results.",
                                "default": 1
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating filter (1.0-5.0)",
                                "default": 4.0
                            }
                        },
                        "required": ["location", "keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_multiple_keywords",
                    "description": "Search for places with MULTIPLE keywords/interests based on user interests and accommodation location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number", "description": "Latitude of accommodation"},
                                    "lng": {"type": "number", "description": "Longitude of accommodation"},
                                    # "neighborhood": {"type": "string", "description": "Neighborhood name (optional)"}
                                },
                                "required": ["lat", "lng"],
                                "description": "Accommodation location with coordinates"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of user interests"
                            },
                            "radius": {
                                "type": "integer",
                                "description": "Search radius in meters (default: 5000, increases if not enough results)",
                                "default": 5000
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum number of pages to retrieve (default: 1), Each page has up to 20 results.",
                                "default": 1
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating filter (1.0-5.0)",
                                "default": 4.0
                            }
                        },
                        "required": ["location", "keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_place_details",
                    "description": "Get comprehensive detailed information about on a list of places",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "place_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Google Places API place_ids"
                            },
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific fields to retrieve",
                                "default": ["name", "formatted_address", "geometry", "opening_hours", "rating", "website", "price_level", "type"]
                            }
                        },
                        "required": ["place_ids"]
                    }
                }
            }
        ]

    def __call__(self, message: str) -> Dict:
        """Execute reasoning and return raw data."""
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": str(result)})
        return result
    
    def execute(self) -> Dict:
        """Execute with function calling support, return raw data."""
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
            
            # Return raw data from tools
            return {
                "reasoning": message.content,
                "tool_results": all_tool_results,
                "raw_places": self._extract_places_from_results(all_tool_results)
            }
        
        return {
            "reasoning": message.content,
            "tool_results": [],
            "raw_places": []
        }
    
    def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute tool call and return raw results."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"REASONING AGENT: Executing {tool_name} with args: {tool_args}")
        
        if tool_name in self.known_actions:
            return self.known_actions[tool_name](**tool_args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _extract_places_from_results(self, tool_results: List[Dict]) -> List[Dict]:
        """Extract all place data from tool results."""
        all_places = []
        for result in tool_results:
            if result["tool"] in ["search_places", "search_multiple_keywords"]:
                places = result.get("result", [])
                if isinstance(places, list):
                    all_places.extend(places)
        return all_places

    def calculate_required_places(self, pace: str, duration_days: int) -> int:
        """
        Agent's internal goal-setting: Calculate required number of places.
        """
        try:
            pace_mapping = {
                "slow": 2,
                "relaxed": 4,
                "moderate": 4,
                "active": 6,
                "standard": 6,
                "fast": 8,
                "intensive": 8
            }
            
            multiplier = float(os.getenv("MAX_PLACES_MULTIPLIER", "2"))
            pace_value = pace_mapping.get(pace.lower(), 6)
            required_places = int(pace_value * duration_days * multiplier)
            
            # Only enforce minimum, no maximum cap
            required_places = max(required_places, 15)  # Minimum 15 places

            print(f"REASONING AGENT: Goal set - {required_places} places for {pace} pace over {duration_days} days")
            return required_places
            
        except (ValueError, TypeError, AttributeError):
            print("REASONING AGENT: Failed to calculate required places, using default: 15")
            return 15
        
# ## Places Research Agent - Formatting Implementation
# - Takes raw Google Places API data as input
# - Transforms it into specified schema format required by the planning agent
# - Sets null values for unavailable data

from tools import search_wikipedia

class PlacesResearchFormattingAgent:
    """
    Specialized agent for formatting raw Google Places API data into structured output.
    Handles data from both search_places/search_multiple_keywords and get_place_details.
    Includes Wikipedia integration.
    """
    
    def __init__(self, use_llm_for_all: bool = False):
        # Initialize OpenAI client for complex formatting
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "gpt-4o"
        self.temperature = 0.1  # Lower temperature for consistent formatting
        self.use_llm_for_all = use_llm_for_all  # Option to use LLM for entire transformation
        
        # Known actions mapping
        self.known_actions = {
            "search_wikipedia": search_wikipedia
        }
        
    def _define_tools(self) -> List[Dict]:
        """Define available tools for the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search Wikipedia for information about a place.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "The name of the place to search for."
                            }
                        },
                        "required": ["search_term"]
                    }
                }
            }
        ]

    def format_response(self, reasoning_data: Dict) -> Dict:
        """
        Format the complete response from reasoning agent.
        
        Args:
            reasoning_data: Raw data from reasoning agent containing:
                - reasoning: The thought process
                - tool_results: List of tool execution results
                - raw_places: Extracted places from tool results
        
        Returns:
            Formatted response with structured place data
        """
        # Extract raw places from the reasoning data
        raw_places = reasoning_data.get('raw_places', [])
        
        # If raw_places is empty, try extracting from tool_results
        if not raw_places:
            tool_results = reasoning_data.get('tool_results', [])
            for tool_result in tool_results:
                if tool_result.get('tool') in ['search_places', 'search_multiple_keywords']:
                    result = tool_result.get('result', [])
                    if isinstance(result, list):
                        raw_places.extend(result)
        
        logger.info(f"Processing {len(raw_places)} raw places for formatting")
        
        # Format the places using the existing format_places method
        formatted_places = self.format_places(raw_places, {})
        
        return {
            "reasoning": reasoning_data.get('reasoning'),
            "places_found": len(formatted_places),
            "formatted_places": formatted_places
        }
    
    def format_places(self, search_results: List[Dict] = None, details_results: Dict[str, Dict] = None) -> List[Dict]:
        """
        Format raw Google Places API data into structured schema.
        
        Args:
            search_results: List of raw place data from search_places/search_multiple_keywords
            details_results: Dict of place_id -> details from get_place_details
            
        Returns:
            List of formatted place dictionaries according to schema
        """
        # Option 1: Use full LLM transformation (if enabled)
        if self.use_llm_for_all:
            return self._format_places_llm_only(search_results, details_results)
        
        # Option 2: Use hybrid approach (default - more efficient)
        formatted_places = []
        
        # Process search results
        if search_results:
            for place in search_results:
                place_id = place.get('place_id')
                
                # Get details if available
                details = details_results.get(place_id, {}) if details_results else {}
                
                # Merge search and details data
                formatted_place = self._format_single_place(place, details)
                formatted_places.append(formatted_place)
        
        # If we only have details results (no search results)
        elif details_results:
            for place_id, details in details_results.items():
                formatted_place = self._format_single_place({}, details)
                formatted_place['place_id'] = place_id
                formatted_places.append(formatted_place)
        
        # Generate descriptions and tags using Wikipedia
        if formatted_places:
            formatted_places = self._generate_wikipedia_descriptions(formatted_places)
        
        return formatted_places
    
    def _format_single_place(self, search_data: Dict, details_data: Dict) -> Dict:
        """
        Format a single place from raw API data to structured schema.
        Combines data from both search and details API responses.
        """
        # Extract place_id and name (prefer from search, fallback to details)
        place_id = search_data.get('place_id') or details_data.get('place_id')
        name = search_data.get('name') or details_data.get('name', 'Unknown Place')
        
        # Extract types and map to standard type
        types = search_data.get('types', []) or details_data.get('types', [])
        primary_type = self._map_to_standard_type(types[0] if types else None)
        
        # Extract geo coordinates
        geo = self._extract_geo(search_data, details_data)
        
        # Calculate geo_cluster_id
        geo_cluster_id = self._calculate_geo_cluster(geo)
        
        # Extract address (prefer details formatted_address, fallback to vicinity)
        address = (details_data.get('formatted_address') or 
            search_data.get('vicinity') or 
            search_data.get('formatted_address'))
        
        # Extract rating (can be in either response)
        rating = details_data.get('rating') or search_data.get('rating')
        
        # Extract cost from price_level if available
        price_level = details_data.get('price_level') or search_data.get('price_level')
        cost_sgd = self._map_price_level_to_cost(price_level)
        
        # Extract website from details
        website = details_data.get('website')
        
        # Extract opening hours for LLM processing
        opening_hours_raw = details_data.get('opening_hours', {})
        
        # Format the place according to schema
        formatted_place = {
            "place_id": place_id,
            "name": name,
            "type": primary_type,
            "cost_sgd": cost_sgd,
            "onsite_co2_kg": None,  # Ignored
            "geo": geo,
            "geo_cluster_id": geo_cluster_id,
            "address": address,
            "nearest_mrt": None,  # Ignored
            "opening_hours": {
                "monday": None,
                "tuesday": None,
                "wednesday": None,
                "thursday": None,
                "friday": None,
                "saturday": None,
                "sunday": None
            },  # Will be formatted by LLM if data available
            "duration_recommended_minutes": None,  # Ignored
            "ticket_price_sgd": {
                "adult": None,
                "child": None,
                "senior": None
            },  # Ignored
            "vegetarian_friendly": None,  # Ignored
            "low_carbon_score": None,  # Ignored
            "description": None,  # Will be set by Wikipedia search
            "links": {
                "official": website,
                "reviews": None  # Ignored
            },
            "rating": rating,
            "tags": [],  # Will be set by Wikipedia search
            "_raw_types": types,  # Store for reference
            "_raw_opening_hours": opening_hours_raw  # Store for reference
        }
        
        return formatted_place
    
    def _calculate_geo_cluster(self, geo: Optional[Dict]) -> Optional[str]:
        """Calculate geo cluster ID from coordinates using Singapore geographic rules."""
        if not geo or not geo.get('latitude') or not geo.get('longitude'):
            return None
            
        lat = geo['latitude']
        lng = geo['longitude']
        
        # Singapore bounds check
        if not (1.1 <= lat <= 1.5 and 103.6 <= lng <= 104.1):
            return None
        
        # Cluster logic for Singapore
        if 1.25 <= lat <= 1.35 and 103.7 <= lng <= 103.9:
            return "central"
        elif lat > 1.35:
            return "north"
        elif lat < 1.35:
            return "south"
        elif lng > 103.8:
            return "east"
        else:
            return "west"
    
    def _generate_wikipedia_descriptions(self, places: List[Dict]) -> List[Dict]:
        """
        Generate descriptions and tags for places using Wikipedia search.
        Process in batches for efficiency.
        """
        if not places:
            return places
        
        # Process in batches to avoid token limits
        batch_size = 10
        processed_places = []
        
        for i in range(0, len(places), batch_size):
            batch = places[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing Wikipedia batch {batch_num} with {len(batch)} places")
            
            # Search Wikipedia for each place
            wikipedia_results = {}
            for place in batch:
                try:
                    wiki_result = search_wikipedia(search_term=place["name"])
                    wikipedia_results[place["name"]] = wiki_result
                    logger.info(f"Found Wikipedia info for: {place['name']}")
                except Exception as e:
                    logger.warning(f"No Wikipedia info for {place['name']}: {e}")
                    wikipedia_results[place["name"]] = None
            
            # Use LLM to create descriptions and tags from Wikipedia results
            try:
                enriched_batch = self._create_descriptions_from_wikipedia(batch, wikipedia_results)
                processed_places.extend(enriched_batch)
                logger.info(f"Successfully enriched batch {batch_num}")
            except Exception as e:
                logger.error(f"Failed to create descriptions for batch {batch_num}: {e}")
                # Keep places without descriptions/tags
                for place in batch:
                    place.pop("_raw_types", None)
                    place.pop("_raw_opening_hours", None)
                processed_places.extend(batch)
        
        return processed_places
    
    def _create_descriptions_from_wikipedia(self, batch: List[Dict], wikipedia_results: Dict) -> List[Dict]:
        """
        Use LLM to create descriptions and tags from Wikipedia search results.
        """
        # Prepare data for LLM
        places_with_wiki = []
        for place in batch:
            place_data = {
                "name": place["name"],
                "type": place["type"],
                "address": place.get("address"),
                "rating": place.get("rating"),
                "wikipedia_info": wikipedia_results.get(place["name"])
            }
            places_with_wiki.append(place_data)
        
        prompt = f"""You are a Singapore tourism expert. Create compelling descriptions and tags for these places based on their Wikipedia information.

Places with Wikipedia data:
{json.dumps(places_with_wiki, indent=2)}

For each place:
1. If Wikipedia info is available, use it to create an accurate, engaging 1-2 sentence description
2. If no Wikipedia info, create a simple description based on the name and type
3. Generate 3-5 relevant tags for travelers

Return a JSON array (no markdown formatting) with the same order, each object containing:
- description: 1-2 sentence compelling tourist description
- tags: array of 3-5 relevant tags

Example format:
[{{"description": "Iconic waterfront park featuring Singapore's famous Merlion statue", "tags": ["landmark", "waterfront", "photo spot", "free entry", "iconic"]}}]
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a Singapore tourism expert. Create descriptions and tags based on Wikipedia information."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        llm_output = response.choices[0].message.content.strip()
        
        # Handle markdown-wrapped JSON
        llm_output = self._extract_json_from_markdown(llm_output)
        
        logger.info(f"LLM response (first 100 chars): {llm_output[:100]}...")
        
        llm_data = json.loads(llm_output)
        
        # Merge LLM data back into places
        for i, place in enumerate(batch):
            if i < len(llm_data):
                llm_place = llm_data[i]
                place["description"] = llm_place.get("description", "A place to visit in Singapore")
                place["tags"] = llm_place.get("tags", [place["type"]])
            else:
                place["description"] = f"A {place['type']} in Singapore"
                place["tags"] = [place["type"]]
            
            # Clean up temp fields
            place.pop("_raw_types", None)
            place.pop("_raw_opening_hours", None)
        
        return batch
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks."""
        import re
        
        # Remove ```json and ``` markers
        if '```json' in text:
            pattern = r'```json\s*(.*?)\s*```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Remove ``` markers without json
        if text.startswith('```') and text.endswith('```'):
            lines = text.split('\n')
            if lines[0] == '```' or lines[0].startswith('```'):
                lines = lines[1:]
            if lines[-1] == '```':
                lines = lines[:-1]
            return '\n'.join(lines).strip()
        
        return text.strip()
    
    def _format_places_llm_only(self, 
                                search_results: List[Dict] = None,
                                details_results: Dict[str, Dict] = None) -> List[Dict]:
        """
        Use LLM for complete transformation (fallback method).
        Less efficient but more flexible for handling unexpected data formats.
        """
        # Combine all data
        combined_data = []
        
        if search_results:
            for place in search_results:
                place_id = place.get('place_id')
                if details_results and place_id in details_results:
                    # Merge search and details
                    merged = {**place, **details_results[place_id]}
                    combined_data.append(merged)
                else:
                    combined_data.append(place)
        elif details_results:
            combined_data = list(details_results.values())
        
        if not combined_data:
            return []
        
        try:
            # Use LLM for complete transformation
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_full_transformation_prompt()},
                    {"role": "user", "content": f"Transform these places:\n{json.dumps(combined_data, indent=2)}"}
                ],
                temperature=self.temperature
            )
            
            result_text = self._extract_json_from_markdown(response.choices[0].message.content)
            result = json.loads(result_text)
            return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"Error in LLM transformation: {e}")
            # Fallback to hybrid approach
            self.use_llm_for_all = False  # Prevent infinite recursion
            result = self.format_places(search_results, details_results)
            self.use_llm_for_all = True  # Restore original setting
            return result
    
    def _extract_geo(self, search_data: Dict, details_data: Dict) -> Optional[Dict]:
        """Extract geo coordinates from either search or details data."""
        # Try search data first
        if search_data.get('geometry'):
            location = search_data['geometry'].get('location', {})
            if location:
                return {
                    "latitude": location.get('lat'),
                    "longitude": location.get('lng')
                }
        
        # Try details data
        if details_data.get('geometry'):
            location = details_data['geometry'].get('location', {})
            if location:
                return {
                    "latitude": location.get('lat'),
                    "longitude": location.get('lng')
                }
        
        return None
    
    def _map_to_standard_type(self, google_type: str) -> str:
        """Map Google Places type to standard type."""
        if not google_type:
            return "attraction"
        
        type_mapping = {
            "tourist_attraction": "attraction",
            "point_of_interest": "attraction",
            "establishment": "attraction",
            "restaurant": "food",
            "food": "food",
            "cafe": "cafe",
            "bar": "bar",
            "bakery": "bakery",
            "park": "park",
            "museum": "museum",
            "shopping_mall": "shopping",
            "lodging": "accommodation",
            "hotel": "accommodation",
            "art_gallery": "museum"
        }
        
        return type_mapping.get(google_type.lower(), "attraction")
    
    def _map_price_level_to_cost(self, price_level: Optional[int]) -> Optional[int]:
        """Map Google's price level (0-4) to cost range."""
        if price_level is None:
            return None
        
        price_mapping = {
            0: 0,  # Free or $0-5
            1: 1,  # $5-15
            2: 2,  # $15-30
            3: 3,  # $30-60
            4: 4   # $60+
        }
        
        return price_mapping.get(price_level, None)
    
    def _get_full_transformation_prompt(self) -> str:
        """
        Full system prompt for complete LLM-based transformation.
        """
        return """
            You are a data transformation assistant. Your task is to take a list of Google Places API JSON objects and convert each object into the following custom attraction format.
            
            - If a field from the custom format is not available in the input data, set it to Python's None. 
            - Use the "types" field from Google Places as "tags" in the output.
            - Use the "geometry.location.lat" and "geometry.location.lng" fields for latitude and longitude.
            - Use the "formatted_address" field for address, fallback to "vicinity" if not available.
            - Map the primary type to standard types (restaurant/food→"food", park→"park", museum/art_gallery→"museum", hotel/lodging→"accommodation", else→"attraction").
            - Opening hours should be formatted as "HH:MM-HH:MM" or "Open 24 hours" from weekday_text.
            - Map price_level (0-4) to cost_sgd (0=free, 1=$5-15, 2=$15-30, 3=$30-60, 4=$60+).
            - Generate engaging descriptions for tourists.
            - Set geo_cluster_id based on coordinates (lat>1.35→north, lat<1.35→south, lng>103.8→east, lng<103.8→west, central if between).
            """.strip()
            
def run_agentic_workflow(input_data):
    """Run agent workflow: coordinate between reasoning and formatting agents."""
    
    # Initialize agents
    reasoning_agent = PlacesResearchReasoningAgent()
    formatting_agent = PlacesResearchFormattingAgent()
    
    # Create prompt for reasoning agent with input data
    prompt = f"""
    I need to find attractions for a trip to Singapore with the following requirements:
    
    Trip Details:
    - Duration: {input_data.get('duration_days', 1)} days
    - Pace: {input_data.get('pace', 'moderate')}
    - Budget: ${input_data.get('budget', 0)}
    
    Optional Preferences:
    - Interests: {input_data.get('optional', {}).get('interests', [])}
    - Accommodation Location: {input_data.get('optional', {}).get('accommodation_location', {})}
    
    Please search for appropriate attractions near the accommodation location based on these preferences.
    """
    
    # Get raw data from reasoning agent
    reasoning_result = reasoning_agent(prompt)
    
    # Format the results using formatting agent
    formatted_result = formatting_agent.format_response(reasoning_result)
    
    return formatted_result

# Load input data
input_file = 'inputs/garden_only_input.json'
input_data = load_input_file(input_file)

result = run_agentic_workflow(input_data)
with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Results written to output.json")