# # Singapore Attraction Discovery Agent
# Agentic system for discovering Singapore attractions using tool-based reasoning, LLM decision-making, and iterative refinement to meet user requirements.

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
    for file in os.listdir('./inputs'):
        if file.endswith('.json'):
            print(f"  ./inputs/{file}")
except FileNotFoundError:
    print("No inputs directory found")

# ## Places Research Agent - Reasoning Implementation
# - Handles pure reasoning and analytical thinking
# - Executes API calls through tools (search_places, search_multiple_keywords, get_place_details)
# - Returns raw data objects from API calls without any formatting
# - Has the calculate_required_places method

from tools import search_places, search_multiple_keywords, get_place_details, search_wikipedia, fetch_place_enrichments, remove_unicode, generate_tags, standardize_opening_hours
from tool_clustering import calculate_geo_cluster
from config import COMMON_INTEREST_MAPPINGS, DIETARY_KEYWORDS
import time

class PlacesResearchAgent:
    """
    Combined agent for researching and formatting Singapore attractions and food places.

    Features:
    - Dictionary-based interest mapping (80% coverage, minimal LLM use)
    - Strict place count enforcement (stops when target met)
    - Dietary-aware food search (Singapore-specific queries)
    - Timeout mechanism (13 minutes hard limit)
    - Integrated formatting (research + format in one class)

    Usage:
        agent = PlacesResearchAgent()
        results = agent.research_and_format(input_data)
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

    def calculate_required_places(self, pace: str, duration_days: int, multiplier: float = None) -> dict:
        """
        Calculate exact number of places needed.

        Formula:
        - Attractions: pace_value * days * multiplier (minimum multiplier = 2)
        - Food: 2 per day (minimum)

        Args:
            pace: Trip pace (slow, relaxed, moderate, active, fast)
            duration_days: Number of days
            multiplier: Optional multiplier override (default from env or 2)

        Returns:
            dict with attractions_needed, food_places_needed, total_needed
        """
        pace_mapping = {
            "slow": 2,
            "relaxed": 3,
            "moderate": 4,
            "active": 5,
            "fast": 6,
        }

        pace_value = pace_mapping.get(pace.lower(), 4)

        # Get multiplier from env or parameter (minimum 2)
        if multiplier is None:
            multiplier = float(os.getenv("MAX_PLACES_MULTIPLIER", "2"))
        multiplier = max(multiplier, 2.0)  # Enforce minimum of 2

        # Attractions calculation: pace * days * multiplier
        attractions_needed = int(pace_value * duration_days * multiplier)

        # Food calculation: 2 per day minimum
        food_per_day = 2
        food_places_needed = food_per_day * duration_days

        requirements = {
            "attractions_needed": attractions_needed,
            "food_places_needed": food_places_needed,
            "total_needed": attractions_needed + food_places_needed,
            "pace_value": pace_value,
            "multiplier": multiplier
        }

        logger.info(f"Requirements: {requirements['attractions_needed']} attractions + {requirements['food_places_needed']} food = {requirements['total_needed']} total")
        return requirements

    def map_interests(self, user_interests: List[str]) -> List[str]:
        """
        Map user interests to Google Places categories using hybrid approach.
        First checks dictionary, then uses LLM for unmapped interests.
        """
        mapped = []
        unmapped = []

        for interest in user_interests:
            interest_lower = interest.lower().strip()

            # Check if it's a dietary keyword
            if interest_lower in DIETARY_KEYWORDS:
                mapped.append(f"{interest_lower} food")
                continue

            # Check dictionary mapping
            if interest_lower in COMMON_INTEREST_MAPPINGS:
                category = COMMON_INTEREST_MAPPINGS[interest_lower]
                # Skip food categories for attractions
                if category not in ["food", "restaurant"]:
                    mapped.append(category)
            else:
                unmapped.append(interest)

        # Use LLM for unmapped interests (batch call)
        if unmapped:
            logger.info(f"Using LLM to map {len(unmapped)} interests: {unmapped}")
            llm_mapped = self._map_interests_with_llm(unmapped)
            mapped.extend(llm_mapped)

        # Default to tourist_attraction if nothing mapped
        if not mapped:
            mapped.append("tourist_attraction")

        logger.info(f"Mapped interests: {mapped}")
        return mapped

    def _map_interests_with_llm(self, unmapped_interests: List[str]) -> List[str]:
        """Use LLM to map unmapped interests to categories."""
        prompt = f"""Map these interests to Google Places categories. Return only valid categories.

Valid categories: tourist_attraction, museum, park, food, cafe, bar, bakery, shopping_mall, lodging

Interests to map: {', '.join(unmapped_interests)}

Return JSON array of categories (e.g., ["museum", "park"])"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You map user interests to Google Places API categories. Return only a JSON array."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content.strip()
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                categories = json.loads(json_match.group())
                return [c for c in categories if c not in ["food", "restaurant"]]
            return ["tourist_attraction"]

        except Exception as e:
            logger.error(f"LLM mapping failed: {e}")
            return ["tourist_attraction"]

    def create_food_search_terms(self, food_interests: List[str], dietary_restrictions: List[str], days: int) -> List[str]:
        """
        Generate food search terms based on food interests and dietary restrictions.
        Prioritizes specific food types (restaurant, cafe) if mentioned in interests.
        """
        base_searches = []

        # Check if user has specific food type interests
        has_restaurant = any('restaurant' in str(i).lower() for i in food_interests)
        has_cafe = any('cafe' in str(i).lower() or 'coffee' in str(i).lower() for i in food_interests)
        has_hawker = any('hawker' in str(i).lower() or 'food court' in str(i).lower() for i in food_interests)

        # Build search terms based on interests
        if has_restaurant:
            # Prioritize restaurants
            if dietary_restrictions:
                for restriction in dietary_restrictions:
                    base_searches.extend([
                        f"{restriction} restaurant Singapore",
                        f"Singapore {restriction} dining"
                    ])
            else:
                base_searches.extend([
                    "restaurant Singapore",
                    "fine dining Singapore",
                    "local restaurant Singapore"
                ])
        elif has_cafe:
            # Prioritize cafes
            base_searches.extend([
                "cafe Singapore",
                "coffee shop Singapore",
                "breakfast cafe Singapore"
            ])
        elif has_hawker:
            # Prioritize hawker centers
            base_searches.extend([
                "hawker centre Singapore",
                "food court Singapore",
                "local hawker Singapore"
            ])
        else:
            # Mixed approach - variety of food places
            if dietary_restrictions:
                for restriction in dietary_restrictions[:2]:  # Limit to 2 restrictions
                    base_searches.append(f"{restriction} food Singapore")
            else:
                base_searches.extend([
                    "restaurant Singapore",
                    "cafe Singapore",
                    "hawker centre Singapore"
                ])

        # Limit searches to avoid over-fetching
        return base_searches[:3]

    def check_timeout(self, start_time: float, timeout_seconds: int = 780) -> bool:
        """Check if execution is approaching timeout (13 minutes = 780 seconds)."""
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            logger.warning(f"Timeout reached: {elapsed:.1f}s")
            return True
        return False

    def search_with_requirements(self, location, keyword, min_rating, max_results_needed, search_type='attraction'):
        """
        Helper to search with progressive radius, pages, and rating relaxation.

        Strategy progressively expands search area AND relaxes rating requirements:
        - Attractions: Start at 4.0 rating, can go down to 3.0 or accept anything
        - Food: Start at 4.5 rating, only go down to 3.8 (maintain food quality)

        Args:
            location: Location dict with lat/lng
            keyword: Search keyword
            min_rating: Initial minimum rating (can be overridden by strategy)
            max_results_needed: Target number of results
            search_type: 'attraction' or 'food' (determines rating relaxation strategy)

        Returns:
            List of places (up to max_results_needed)
        """
        # Progressive search strategies
        attraction_strategies = [
            {'radius': 5000,  'pages': 1, 'min_rating': 4.0},  # Start strict
            {'radius': 7500,  'pages': 2, 'min_rating': 3.9},  # Small relaxation
            {'radius': 10000, 'pages': 2, 'min_rating': 3.8},  # Gradual decline
            {'radius': 12500, 'pages': 2, 'min_rating': 3.7},  # Mid-point
            {'radius': 15000, 'pages': 3, 'min_rating': 3.5},  # Accept decent places
            {'radius': 17500, 'pages': 3, 'min_rating': 3.3},  # Getting desperate
            {'radius': 20000, 'pages': 3, 'min_rating': 3.0},  # Most of SG
            {'radius': 25000, 'pages': 3, 'min_rating': 2.5},  # Nearly all SG
            {'radius': 30000, 'pages': 3, 'min_rating': 2.5},  # Entire island
            {'radius': 35000, 'pages': 3, 'min_rating': 2.5},  # Include offshore islands
        ]

        food_strategies = [
            {'radius': 5000,  'pages': 1, 'min_rating': 4.5},  # Start very strict
            {'radius': 7500,  'pages': 2, 'min_rating': 4.4},  # Slight relaxation
            {'radius': 10000, 'pages': 2, 'min_rating': 4.3},  # Still high quality
            {'radius': 12500, 'pages': 2, 'min_rating': 4.2},  # Very good food
            {'radius': 15000, 'pages': 3, 'min_rating': 4.1},  # Above 4.0
            {'radius': 17500, 'pages': 3, 'min_rating': 4.0},  # Good threshold
            {'radius': 20000, 'pages': 3, 'min_rating': 3.9},  # Most of SG
            {'radius': 25000, 'pages': 3, 'min_rating': 3.8},  # Nearly all SG
            {'radius': 30000, 'pages': 3, 'min_rating': 3.7},  # Entire island
            {'radius': 35000, 'pages': 3, 'min_rating': 3.5},  # Absolute minimum
        ]

        # Select strategy based on search type
        strategies = food_strategies if search_type == 'food' else attraction_strategies

        all_results = []

        for level, strategy in enumerate(strategies, 1):
            if len(all_results) >= max_results_needed:
                print(f"Target reached: {len(all_results)}/{max_results_needed} for '{keyword}'")
                break

            radius = strategy['radius']
            max_pages = strategy['pages']
            current_min_rating = strategy['min_rating']

            still_needed = max_results_needed - len(all_results)
            logger.info(f"[Level {level}/{len(strategies)}] Searching '{keyword}' at radius={radius}m, pages={max_pages}, rating>={current_min_rating} (need {still_needed} more)")
            print(f"  [{level}/{len(strategies)}] {keyword}: {radius/1000}km, {max_pages}pg, rating>={current_min_rating:.1f}...")

            results = search_places(
                location=location,
                keyword=keyword,
                radius=radius,
                max_pages=max_pages,
                min_rating=current_min_rating
            )

            # Deduplicate by place_id
            existing_ids = {p.get('place_id') for p in all_results}
            new_results = [p for p in results if p.get('place_id') not in existing_ids]

            all_results.extend(new_results)
            logger.info(f"Found {len(new_results)} new results at level {level} (total: {len(all_results)})")
            print(f"      -> +{len(new_results)} new (total: {len(all_results)}/{max_results_needed})")

            # If we got enough, stop expanding
            if len(all_results) >= max_results_needed:
                print(f"Target reached for '{keyword}'")
                break

            # If no new results at current level, continue to next level
            if not new_results:
                print(f"      WARNING: No new results, trying next level...")
                continue

        # Return up to max_results_needed
        final_count = min(len(all_results), max_results_needed)
        if final_count < max_results_needed:
            print(f"WARNING: Only found {final_count}/{max_results_needed} for '{keyword}' (type: {search_type})")

        return all_results[:max_results_needed]

    # === FORMATTING METHODS ===

    def format_results(self, raw_places: List[Dict], enrichments: Dict = None) -> List[Dict]:
        """
        Format raw places from search into structured schema.
        This is the only formatting method - takes raw results and formats them.

        Args:
            raw_places: List of raw place data from search API
            enrichments: Optional dict with 'details' and 'wikipedia' data
            from fetch_place_enrichments()

        Returns:
            List of formatted place dictionaries
        """
        if not enrichments:
            enrichments = {'details': {}, 'wikipedia': {}}

        details_by_id = enrichments.get('details', {})
        wikipedia_by_name = enrichments.get('wikipedia', {})

        formatted_places = []
        for place in raw_places:
            place_id = place.get('place_id')
            place_name = place.get('name')

            # Get enrichment data for this place
            details_data = details_by_id.get(place_id)
            wikipedia_summary = wikipedia_by_name.get(place_name)

            # Format the place with enriched data
            formatted_place = self._format_single_place(place, details_data, wikipedia_summary)
            formatted_places.append(formatted_place)

        return formatted_places

    def _format_single_place(self, place_data: Dict, details_data: Dict = None, wikipedia_summary: str = None) -> Dict:
        """
        Format single place from raw API data to structured schema.

        Args:
            place_data: Raw place data from search API
            details_data: Optional detailed data from get_place_details
            wikipedia_summary: Optional Wikipedia summary
        """
        # Merge details if available
        if details_data:
            place_data = {**place_data, **details_data}

        # Extract geo coordinates
        geo = self._extract_geo(place_data)

        # Calculate geo cluster ID
        geo_cluster_id = None
        if geo and geo.get('latitude') and geo.get('longitude'):
            geo_cluster_id = calculate_geo_cluster(geo['latitude'], geo['longitude'])

        # Extract opening hours if available
        # NOTE: If no opening hours are provided by the API, we default to "00:00-23:59" (open all day)
        # This ensures that places without explicit hours are still usable in the itinerary
        opening_hours = {
            "monday": None,
            "tuesday": None,
            "wednesday": None,
            "thursday": None,
            "friday": None,
            "saturday": None,
            "sunday": None
        }
        if place_data.get('opening_hours'):
            opening_hours = self._parse_opening_hours(place_data['opening_hours'])
        else:
            # Default to open all day if no hours provided
            default_hours = "00:00-23:59"
            opening_hours = {
                "monday": default_hours,
                "tuesday": default_hours,
                "wednesday": default_hours,
                "thursday": default_hours,
                "friday": default_hours,
                "saturday": default_hours,
                "sunday": default_hours
            }

        # Extract reviews count (user_ratings_total)
        reviews_count = place_data.get('user_ratings_total')

        # Extract accessibility options
        accessibility_options = []
        if place_data.get('wheelchair_accessible_entrance'):
            accessibility_options.append("wheelchair_accessible_entrance")

        # Get raw text fields
        name = place_data.get('name', 'Unknown Place')
        address = place_data.get('formatted_address') or place_data.get('vicinity')
        all_types = place_data.get('types', [])
        place_type = self._map_to_standard_type(all_types)
        description = wikipedia_summary or f"A {place_type} in Singapore"

        # Generate enhanced tags using tools function
        tags = generate_tags(
            place_type=place_type,
            all_types=all_types,
            accessibility_options=accessibility_options,
            price_level=place_data.get('price_level'),
            rating=place_data.get('rating'),
            description=description,
            name=name,
            reviews_count=reviews_count,
            openai_client=self.client
        )

        return {
            "place_id": place_data.get('place_id'),
            "name": remove_unicode(name),
            "type": place_type,
            "cost_sgd": self._map_price_level_to_cost(place_data.get('price_level')),
            "onsite_co2_kg": None,
            "geo": geo,
            "geo_cluster_id": geo_cluster_id,
            "address": remove_unicode(address) if address else None,
            "nearest_mrt": None,
            "opening_hours": opening_hours,
            "duration_recommended_minutes": None,
            "ticket_price_sgd": {
                "adult": None,
                "child": None,
                "senior": None
            },
            "accessibility_options": accessibility_options,
            "low_carbon_score": None,
            "description": remove_unicode(description),
            "links": {"official": place_data.get('website'), "reviews": reviews_count},
            "rating": place_data.get('rating'),
            "tags": tags
        }

    def _extract_geo(self, place_data: Dict) -> Optional[Dict]:
        """Extract geo coordinates from place data."""
        geometry = place_data.get('geometry')
        if not geometry:
            return None
        location = geometry.get('location', {})
        return {"latitude": location.get('lat'), "longitude": location.get('lng')}

    def _map_to_standard_type(self, google_types) -> str:
        """
        Map Google Places types to standard type.
        Prioritizes specific types over generic ones.

        Args:
            google_types: Either a string (single type) or list of types from Google API

        Returns:
            Standard type string
        """
        # Handle single string input (backwards compatibility)
        if isinstance(google_types, str):
            google_types = [google_types]

        if not google_types:
            return "attraction"

        # Type mapping with priorities
        # Priority 1: Food-related (most specific)
        food_types = {
            "restaurant": "food",
            "food": "food",
            "meal_delivery": "food",
            "meal_takeaway": "food",
            "cafe": "cafe",
            "bar": "bar",
            "bakery": "bakery",
            "night_club": "bar"
        }

        # Priority 2: Specific attractions
        attraction_types = {
            "park": "park",
            "museum": "museum",
            "art_gallery": "museum",
            "tourist_attraction": "attraction",
            "amusement_park": "attraction",
            "aquarium": "attraction",
            "zoo": "attraction"
        }

        # Priority 3: Shopping and accommodation
        other_specific = {
            "shopping_mall": "shopping",
            "lodging": "accommodation",
            "hotel": "accommodation"
        }

        # Priority 4: Generic (lowest priority)
        generic_types = {
            "establishment": "attraction",
            "point_of_interest": "attraction"
        }

        # Check types in priority order
        for gtype in google_types:
            gtype_lower = gtype.lower()

            # Check food types first (highest priority)
            if gtype_lower in food_types:
                return food_types[gtype_lower]

        for gtype in google_types:
            gtype_lower = gtype.lower()

            # Check specific attraction types
            if gtype_lower in attraction_types:
                return attraction_types[gtype_lower]

        for gtype in google_types:
            gtype_lower = gtype.lower()

            # Check other specific types
            if gtype_lower in other_specific:
                return other_specific[gtype_lower]

        for gtype in google_types:
            gtype_lower = gtype.lower()

            # Check generic types last
            if gtype_lower in generic_types:
                return generic_types[gtype_lower]

        # Default fallback
        return "attraction"

    def _map_price_level_to_cost(self, price_level: Optional[int]) -> Optional[int]:
        """Map Google's price level (0-4) to SGD cost estimate."""
        if price_level is None:
            return None
        return {0: 0, 1: 15, 2: 30, 3: 60, 4: 100}.get(price_level)

    def _parse_opening_hours(self, opening_hours_data: Dict) -> Dict:
        """
        Parse opening hours from Google API response and standardize to 24-hour format.

        Args:
            opening_hours_data: Opening hours dict from Google Places API

        Returns:
            Dict with day names as keys and standardized time strings as values
        """
        result = {
            "monday": None,
            "tuesday": None,
            "wednesday": None,
            "thursday": None,
            "friday": None,
            "saturday": None,
            "sunday": None
        }

        # Google API provides weekday_text as a list like:
        # ["Monday: 9:00 AM – 5:00 PM", "Tuesday: 9:00 AM – 5:00 PM", ...]
        weekday_text = opening_hours_data.get('weekday_text', [])

        day_mapping = {
            'monday': 'monday',
            'tuesday': 'tuesday',
            'wednesday': 'wednesday',
            'thursday': 'thursday',
            'friday': 'friday',
            'saturday': 'saturday',
            'sunday': 'sunday'
        }

        for entry in weekday_text:
            # Entry format: "Monday: 9:00 AM – 5:00 PM" or "Monday: Closed"
            if ':' in entry:
                day_part, hours_part = entry.split(':', 1)
                day_name = day_part.strip().lower()
                hours = hours_part.strip()

                if day_name in day_mapping:
                    # Remove unicode and standardize to 24-hour format
                    hours = remove_unicode(hours)
                    hours = standardize_opening_hours(hours)
                    result[day_mapping[day_name]] = hours

        # If no hours were parsed, default all days to open all day
        if all(v is None for v in result.values()):
            default_hours = "00:00-23:59"
            for day in result:
                result[day] = default_hours

        return result


# ## Places Research Agent - Formatting Implementation (DEPRECATED)
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

class PlacesClusteringAgent:
    """
    Agent for clustering attractions geographically and thematically for optimal travel planning.
    Groups places by geo_cluster_id and calculates travel connections with accommodation.
    """

    def __init__(self):
        self.accommodation_location = None
        self.cluster_mapping = {
            "central": {"name": "Central Singapore"},
            "north": {"name": "Northern Singapore"},
            "south": {"name": "Southern Singapore"},
            "east": {"name": "Eastern Singapore"},
            "west": {"name": "Western Singapore"}
        }

    def cluster_places(self, places_data: Dict, accommodation_location: Dict) -> Dict:
        """
        Cluster places geographically and add travel connections.

        Args:
            places_data: Formatted places data from PlacesResearchFormattingAgent
            accommodation_location: Dict with lat, lng of accommodation

        Returns:
            Clustered data with travel connections and modes
        """
        self.accommodation_location = accommodation_location
        places = places_data.get('formatted_places', [])

        if not places:
            return self._create_empty_clusters_response(places_data)

        # Group places by geo_cluster_id
        clusters = self._group_by_geo_cluster(places)

        # Calculate travel connections for each cluster
        clustered_results = []
        for cluster_id, cluster_places in clusters.items():
            cluster_data = self._create_cluster_data(cluster_id, cluster_places)
            clustered_results.append(cluster_data)

        # Sort clusters by priority (central first)
        clustered_results.sort(key=lambda x: self.cluster_mapping.get(x['cluster_id'], {}).get('priority', 999))

        return {
            "reasoning": places_data.get('reasoning'),
            "places_found": places_data.get('places_found', len(places)),
            "total_clusters": len(clustered_results),
            "accommodation": {
                "location": accommodation_location,
                "geo_cluster_id": self._calculate_geo_cluster_for_coords(
                    accommodation_location.get('lat'),
                    accommodation_location.get('lng') or accommodation_location.get('lon')
                )
            },
            "clustered_places": clustered_results
        }

    def _group_by_geo_cluster(self, places: List[Dict]) -> Dict[str, List[Dict]]:
        """Group places by their geo_cluster_id."""
        clusters = {}
        for place in places:
            cluster_id = place.get('geo_cluster_id', 'unknown')
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(place)
        return clusters

    def _create_cluster_data(self, cluster_id: str, places: List[Dict]) -> Dict:
        """Create cluster data with travel connections."""
        cluster_info = self.cluster_mapping.get(cluster_id, {"name": f"Unknown Cluster ({cluster_id})", "priority": 999})

        # Calculate travel connections for each place
        places_with_travel = []
        for place in places:
            place_with_travel = place.copy()
            travel_info = self._calculate_travel_to_place(place)
            place_with_travel['travel_from_accommodation'] = travel_info
            places_with_travel.append(place_with_travel)

        # Sort places within cluster by distance from accommodation
        places_with_travel.sort(key=lambda x: x['travel_from_accommodation']['distance_km'])

        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_info["name"],
            "places_count": len(places_with_travel),
            "average_distance_km": round(sum(p['travel_from_accommodation']['distance_km'] for p in places_with_travel) / len(places_with_travel), 2),
            "recommended_travel_mode": self._get_cluster_recommended_mode(places_with_travel),
            "places": places_with_travel
        }

    def _calculate_travel_to_place(self, place: Dict) -> Dict:
        """Calculate travel information from accommodation to place."""
        place_geo = place.get('geo', {})
        place_lat = place_geo.get('latitude')
        place_lng = place_geo.get('longitude')

        if not place_lat or not place_lng or not self.accommodation_location:
            return self._create_default_travel_info()

        # Calculate distance using Haversine formula
        # Handle both 'lng' and 'lon' formats for accommodation location
        acc_lng = self.accommodation_location.get('lng') or self.accommodation_location.get('lon')
        distance_km = self._haversine_distance(
            self.accommodation_location.get('lat'),
            acc_lng,
            place_lat,
            place_lng
        )

        # Determine travel modes and times
        travel_modes = self._calculate_travel_modes(distance_km)

        return {
            "distance_km": round(distance_km, 2),
            "travel_modes": travel_modes,
            "recommended_mode": self._get_recommended_mode(distance_km)
        }

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in kilometers
        earth_radius_km = 6371
        return earth_radius_km * c

    def _get_cluster_recommended_mode(self, places: List[Dict]) -> str:
        """Get recommended mode for entire cluster based on distances."""
        if not places:
            return "MRT"

        avg_distance = sum(p['travel_from_accommodation']['distance_km'] for p in places) / len(places)
        return self._get_recommended_mode(avg_distance)

    def _calculate_geo_cluster_for_coords(self, lat: float, lng: float) -> str:
        """Calculate geo cluster for given coordinates."""
        if not lat or not lng:
            return "unknown"

        # Singapore bounds check
        if not (1.1 <= lat <= 1.5 and 103.6 <= lng <= 104.1):
            return "unknown"

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

    def _create_empty_clusters_response(self, places_data: Dict) -> Dict:
        """Create response when no places to cluster."""
        return {
            "reasoning": places_data.get('reasoning'),
            "places_found": 0,
            "total_clusters": 0,
            "accommodation": {
                "location": self.accommodation_location,
                "geo_cluster_id": "unknown"
            },
            "clustered_places": []
        }

# ## Simple Function Handler

def research_places(input_file: str, output_file: str = None) -> dict:
    """
    Simple function handler to research and format Singapore places.

    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results (default: ResearchAgent/output.json)

    Returns:
        Dictionary with formatted places

    Example:
        results = research_places('inputs/simple_input.json')
    """
    import time
    from datetime import datetime

    # Load input
    input_data = load_input_file(input_file)
    if not input_data:
        return {"error": "Failed to load input file"}

    # Initialize agent
    agent = PlacesResearchAgent()
    start_time = time.time()

    # Extract parameters
    pace = input_data.get('pace', 'moderate')
    duration_days = input_data.get('duration_days', 1)
    location = input_data.get('optional', {}).get('accommodation_location', {})
    user_interests = input_data.get('optional', {}).get('interests', [])
    dietary_restrictions = input_data.get('optional', {}).get('dietary_restrictions', [])
    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [dietary_restrictions]

    # Calculate requirements
    requirements = agent.calculate_required_places(pace, duration_days)
    print(f"Requirements: {requirements['attractions_needed']} attractions + {requirements['food_places_needed']} food = {requirements['total_needed']} total")

    # Map interests
    mapped_interests = agent.map_interests(user_interests) if user_interests else ['tourist_attraction']
    print(f"Mapped interests: {mapped_interests}")

    # Search attractions
    print(f"\n=== Searching for Attractions ===")
    attraction_results = []
    for interest in mapped_interests:
        if agent.check_timeout(start_time):
            break
        if len(attraction_results) >= requirements['attractions_needed']:
            print(f"Reached target: {len(attraction_results)}/{requirements['attractions_needed']} attractions")
            break

        min_rating = 3.5 if interest == 'shopping_mall' else 4.0
        results = agent.search_with_requirements(
            location, interest, min_rating,
            requirements['attractions_needed'] - len(attraction_results),
            search_type='attraction'
        )
        attraction_results.extend(results)
        print(f"  Total attractions so far: {len(attraction_results)}/{requirements['attractions_needed']}\n")

    # Fallback mechanism: if still not enough results, try broader keywords
    if len(attraction_results) < requirements['attractions_needed']:
        fallback_keywords = ['attraction', 'establishment']
        print(f"\n=== Insufficient results ({len(attraction_results)}/{requirements['attractions_needed']}). Trying fallback keywords ===")

        for fallback_keyword in fallback_keywords:
            if agent.check_timeout(start_time):
                break
            if len(attraction_results) >= requirements['attractions_needed']:
                print(f"Reached target with fallback: {len(attraction_results)}/{requirements['attractions_needed']} attractions")
                break

            print(f"Trying fallback keyword: '{fallback_keyword}'")
            results = agent.search_with_requirements(
                location, fallback_keyword, 4.0,
                requirements['attractions_needed'] - len(attraction_results),
                search_type='attraction'
            )

            # Deduplicate with existing results
            existing_ids = {p.get('place_id') for p in attraction_results}
            new_results = [p for p in results if p.get('place_id') not in existing_ids]

            attraction_results.extend(new_results)
            print(f"  Added {len(new_results)} new places with '{fallback_keyword}' (total: {len(attraction_results)}/{requirements['attractions_needed']})\n")

    # Search food - extract food-related interests from user interests
    food_interests = [i for i in user_interests if any(
        food_kw in str(i).lower() for food_kw in ['restaurant', 'cafe', 'food', 'dining', 'hawker', 'bar', 'bakery']
    )]

    food_search_terms = agent.create_food_search_terms(food_interests, dietary_restrictions, duration_days)
    print(f"Food interests: {food_interests}")
    print(f"Food search terms: {food_search_terms}")

    print(f"\n=== Searching for Food Places ===")
    food_results = []
    for food_term in food_search_terms:
        if agent.check_timeout(start_time):
            break
        if len(food_results) >= requirements['food_places_needed']:
            print(f"Reached target: {len(food_results)}/{requirements['food_places_needed']} food places")
            break

        # Use rating 4.5+ for food to start (will be relaxed by strategy)
        min_food_rating = 4.5
        results = agent.search_with_requirements(
            location, food_term, min_food_rating,
            requirements['food_places_needed'] - len(food_results),
            search_type='food'
        )
        food_results.extend(results)
        print(f"  Total food places so far: {len(food_results)}/{requirements['food_places_needed']}\n")

    # Combine all places
    all_places = attraction_results + food_results
    print(f"\n=== Total places collected: {len(all_places)} ===")

    # Fetch enrichments (details + Wikipedia) concurrently
    print(f"\n=== Fetching enrichments for all places ===")
    enrichments = fetch_place_enrichments(all_places, fetch_details=True, fetch_wikipedia=True)

    # Format places with enriched data
    print(f"\n=== Formatting {len(all_places)} places ===")
    formatted_places = agent.format_results(all_places, enrichments)

    elapsed = time.time() - start_time

    # Generate retrieval ID with timestamp
    retrieval_id = datetime.now().strftime("ret_%Y_%m_%d_%H%M%S")

    result = {
        "requirements": {
            "destination_city": input_data.get("destination_city", "Singapore"),
            "trip_dates": input_data.get("trip_dates", {}),
            "duration_days": input_data.get("duration_days"),
            "budget": input_data.get("budget"),
            "pace": input_data.get("pace"),
            "optional": input_data.get("optional", {})
        },
        "retrieval": {
            "retrieval_id": retrieval_id,
            "time_elapsed": round(elapsed, 2),
            "places_found": len(formatted_places),
            "attractions_count": len(attraction_results),
            "food_count": len(food_results),
            "conditions": requirements,
            "places_matrix": {
                "nodes": formatted_places
            }
        }
    }

    # Save if output specified
    if not output_file:
        output_file = 'ResearchAgent/output.json'

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary: Found {len(attraction_results)} attractions + {len(food_results)} food = {len(formatted_places)} total places")
    print(f"Time elapsed: {elapsed:.2f}s")
    logger.info(f"Found {len(formatted_places)} places, saved to {output_file}")
    return result


# ## Main Execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        result = research_places(input_file, output_file)
        print(f"Found {result['retrieval']['places_found']} places")
    else:
        print("Usage: python main.py <input_file> [output_file]")
        print("Example: python main.py inputs/input.json outputs/output.json")