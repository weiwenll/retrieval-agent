# # Singapore Attraction Discovery Agent
# Agentic system for discovering Singapore attractions using tool-based reasoning, LLM decision-making, and iterative refinement to meet user requirements.

# Install requirements if needed
# !pip install -r requirements.txt

# Imports
import os
import sys
import json
import logging
import time
from datetime import datetime
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

# Import tools and utilities
from tools import (
    search_places, get_place_details, remove_unicode, standardize_opening_hours, generate_tags,
    convert_dietary_to_exclusions, generate_place_description,
    geocode_location, reverse_geocode, map_interest_to_place_types
)
from tool_clustering import calculate_geo_cluster
from singapore_onsite_carbon_score import get_place_carbon_details

# Import config constants
from config import (
    FOOD_PLACE_TYPES,
    SINGAPORE_GEOCLUSTERS_BOUNDARIES,
    SINGAPORE_GEOCLUSTERS_POINTS,
    ATTRACTION_MULTIPLIER,
    FOOD_MULTIPLIER,
    INTEREST_AUTO_EXCLUSIONS,
    FOOD_SEARCH_PARAMS_BY_CLUSTER,
    DIETARY_EXCLUSIONS
)
    
# Configure logging
handlers = [logging.StreamHandler()]

# Add file handler only when running locally (not in Lambda)
if not os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
    handlers.append(logging.FileHandler('agent_reasoning.log', mode='a'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check API keys
print("API Keys Status:")
print(f"GOOGLE_MAPS_API_KEY: {'Set' if os.getenv('GOOGLE_MAPS_API_KEY') else 'Missing'}")
print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Missing'}")

# ## Input Data Loading
# Load and validate input file with trip requirements.

def load_input_file(file_path: str) -> dict:
    """Load and validate input file."""
    try:
        with open(file_path, 'r') as f:
            input_data = json.load(f)

        # Check if requirements section exists
        if "requirements" not in input_data:
            raise ValueError("Missing required section: requirements")

        requirements = input_data["requirements"]

        # Basic validation - check fields in requirements section
        required_fields = ["trip_dates", "duration_days", "budget_total_sgd", "pace"]
        for field in required_fields:
            if field not in requirements:
                raise ValueError(f"Missing required field in requirements: {field}")

        # Check if optional section exists in requirements
        if "optional" in requirements and "accommodation_location" not in requirements["optional"]:
            raise ValueError("Missing required field: requirements.optional.accommodation_location")

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
# - Executes API calls through tools (search_places, get_place_details)
# - Returns raw data objects from API calls without any formatting
# - Has the calculate_required_places method

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

    def __init__(self, system_prompt="", num_travelers: int = 1):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Agent configuration
        self.model_name = "gpt-4o"
        self.temperature = 0.3
        self.num_travelers = num_travelers  # Total travelers (adults + children) for carbon calculation

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
            "get_place_details": get_place_details,
            "evaluate_results_quality": lambda user_interests, user_uninterests, places=None: self.evaluate_results_quality(
                places if places is not None else self.current_results,
                user_interests,
                user_uninterests,
                self._extract_coordinates(places if places is not None else self.current_results)
            ),
            "map_interest_to_place_types": lambda interest: {
                "interest": interest,
                "place_types": map_interest_to_place_types(interest, self.client)
            }
        }

    def _extract_coordinates(self, places: List[Dict]) -> List[Dict]:
        """Extract coordinates with geo clusters for geographic analysis."""
        coordinates = []
        for place in places:
            location = place.get('location', {})
            lat = location.get('latitude')
            lng = location.get('longitude')

            # Use existing calculate_geo_cluster function
            cluster_id = calculate_geo_cluster(lat, lng) if lat and lng else 'unknown'

            name = place.get('displayName', {})
            if isinstance(name, dict):
                name = name.get('text', 'Unknown')
            else:
                name = place.get('name', 'Unknown')

            coordinates.append({
                'name': name,
                'lat': lat,
                'lng': lng,
                'cluster': cluster_id
            })

        return coordinates

    def _is_food_place(self, place: Dict) -> bool:
        """
        Determine if a place is food-related based on types.
        Uses FOOD_PLACE_TYPES from config.py for comprehensive classification.

        Args:
            place: Place dictionary with 'types' or 'type' field

        Returns:
            True if place is food-related, False otherwise
        """
        # Check types array (new format)
        place_types = set(place.get('types', []))

        # Check single type field (legacy format)
        if not place_types and place.get('type'):
            place_types = {place.get('type')}

        return bool(FOOD_PLACE_TYPES & place_types)

        # Agent state for tracking requirements
        self.required_places = 0
        self.current_results_count = 0
    
    def _get_default_system_prompt(self) -> str:
        """ReAct system prompt for attraction search only (food handled separately via deterministic geo-cluster search)."""
        return """
You are an intelligent Singapore attractions research agent. You use ReAct (Reasoning + Acting) to adaptively search for attraction places only.

## YOUR FOCUS
Search ONLY for attraction places (museums, parks, temples, tourist attractions, etc.)
DO NOT search for food places (restaurants, cafes, bars) - food is handled separately via deterministic geo-cluster search.

## REACT LOOP
Run in cycles of: THOUGHT → ACTION → OBSERVATION → (repeat until requirements met)

## 4 CORE PATTERNS

### Pattern 1: Requirements Analysis
THOUGHT: Analyze user trip (pace, duration, interests, uninterests)
ACTION: Calculate exact needs (attractions count, geo-diversity)
OBSERVATION: Establish search targets and constraints

### Pattern 2: Adaptive Search with Quality Evaluation
THOUGHT: Execute targeted search based on current gaps
ACTION: search_places with specific types, radius, ratings
OBSERVATION: Evaluate batch results with evaluate_results_quality
THOUGHT: Assess diversity, relevance, geographic spread scores
ACTION: Adapt strategy (expand radius, new types, lower ratings)
OBSERVATION: Progressive refinement until targets met

### Pattern 3: Interest Classification & Strategy Selection
THOUGHT: Classify interests into location-based vs category-based
- Location: "near Changi Airport", "around Orchard Road"
- Category: "museums", "temples", "outdoor activities"
ACTION: Prioritize location searches first, then categories
OBSERVATION: Ensure location-specific requests honored
THOUGHT: Respect user uninterests - exclude place types the user dislikes

### Pattern 4: Deduplication with Intelligent Backfill
THOUGHT: Check for duplicate attractions
ACTION: Identify gaps in diversity (experience variety, categories)
OBSERVATION: Missing categories found
ACTION: Targeted backfill search for specific gaps
OBSERVATION: Balanced, diverse final set

## AVAILABLE ACTIONS

1. **map_interest_to_place_types**(interest)
- Map user interest/category to valid Google Place types
- Use at START to convert user interests to search types
- Use DURING SEARCH when you need more diversity or relevance
- Example: If missing "historical place" → map_interest_to_place_types("historical place")
- Returns 1-3 valid Google Place types from ATTRACTION_PLACE_TYPES

2. **search_places**(location, included_types, radius, min_rating, max_results, excluded_types)
- Search Google Places API for ATTRACTIONS ONLY
- Use types from map_interest_to_place_types output
- Use progressive expansion: start radius=10000m, rating=4.0
- Expand radius OR lower rating if needed
- Use excluded_types to filter out uninterests and non-attraction types

3. **evaluate_results_quality**(places, user_interests, user_uninterests)
- LLM evaluates diversity (0-10), relevance (0-10), geographic spread (0-10)
- Returns missing_categories and recommendation
- Use this to identify gaps, then call map_interest_to_place_types to fill them

## SEARCH GUIDELINES

**Geographic Distribution:**
- Singapore has 7 main clusters: North, North East, East, South, West, Central, Downtown
- "Near [location]" = 60% from that area + 40% from other clusters for diversity
- "Around [location]" = 50% local + 50% broader
- No location modifier = Search across all Singapore clusters
- Aim for representation from multiple clusters (avoid 100% concentration in one area)

**Ratings:**
- Shopping malls: min 3.5, floor 3.0
- General attractions: min 4.0, floor 3.0

**Stopping Conditions:**
- Target: {attractions_needed} attraction places
- Acceptable range: 90-110% of target
- Hard stop: 120% of target maximum
- Quality threshold: If within range AND attraction requirements met, stop

**Strategy:**
1. Use map_interest_to_place_types to convert user interests to valid Google types
2. Start narrow (user interests, high ratings, small radius)
3. Evaluate quality after each batch (diversity, relevance, geography)
4. If gaps detected, use map_interest_to_place_types for missing categories
5. Ensure geographic spread across clusters
6. Respect uninterests by excluding unwanted place types
7. Adapt intelligently (expand radius/lower rating/new types)
8. Stop when requirements met (don't overshoot)

**Important:**
- ALWAYS use map_interest_to_place_types to get valid Google Place types
- DO NOT make up place types (e.g., "buddhist_temple" doesn't exist - use place_of_worship)
- Let map_interest_to_place_types handle the mapping to valid types

Return raw place data from API. Focus on meeting requirements with high-quality, diverse, geographically spread results.
""".strip()

    def _define_places_tools(self) -> List[Dict]:
        """Define Google Places API v1 tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_places",
                    "description": "Search for places using Google Places API v1. Supports single or multiple place types with exclusions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "lat": {"type": "number", "description": "Latitude of accommodation"},
                                    "lng": {"type": "number", "description": "Longitude of accommodation"},
                                    "neighborhood": {"type": "string", "description": "Neighborhood name (optional, future: can geocode to lat/lng)"}
                                },
                                "required": ["lat", "lng"],
                                "description": "Accommodation location with coordinates"
                            },
                            "included_types": {
                                "description": "Single type (string) or multiple types (array) to search for",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}
                                ]
                            },
                            "excluded_types": {
                                "description": "Array of place types to exclude (optional)",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}
                                ]
                            },
                            "radius": {
                                "type": "integer",
                                "description": "Search radius in meters (default: 10000, max: 35000)",
                                "default": 10000
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results to retrieve (default: 20, max: 20 per request)",
                                "default": 20
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating filter (1.0-5.0)",
                                "default": 4.0
                            }
                        },
                        "required": ["location", "included_types"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_results_quality",
                    "description": "Evaluate search results quality using LLM to assess diversity, relevance, and geographic spread. Use this after collecting a batch of results to decide next search strategy. If places is not provided, will use current search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_interests": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "User's stated interests for comparison"
                            },
                            "user_uninterests": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "User's stated uninterests/dislikes to avoid"
                            },
                            "places": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Optional list of places to evaluate. If not provided, uses current search results."
                            }
                        },
                        "required": ["user_interests", "user_uninterests"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "map_interest_to_place_types",
                    "description": "Map a user interest/category to 1-3 valid Google Place types using ATTRACTION_PLACE_TYPES. Use this when you need to convert user interests (e.g., 'museums', 'temples', 'nature') into specific Google Place types for searching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "interest": {
                                "type": "string",
                                "description": "User interest or category to map (e.g., 'museums', 'temples', 'parks')"
                            }
                        },
                        "required": ["interest"]
                    }
                }
            }
        ]

    def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute tool call and return raw results."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"REASONING AGENT: Executing {tool_name} with args: {tool_args}")
        
        if tool_name in self.known_actions:
            return self.known_actions[tool_name](**tool_args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    # ========== REACT ORCHESTRATION METHODS ==========

    def set_context(self, context: Dict):
        """Set agent context with user requirements and constraints."""
        self.context = context
        self.current_results = []
        self.start_time = context.get('start_time', time.time())

    def check_timeout(self, max_seconds: int = 780) -> bool:
        """Check if we've exceeded timeout (default 13 minutes)."""
        if not hasattr(self, 'start_time'):
            # If start_time not set, initialize it now
            self.start_time = time.time()
            return False
        elapsed = time.time() - self.start_time
        return elapsed > max_seconds

    def requirements_met(self) -> bool:
        """
        Check if search requirements are satisfied.
        Uses smart stopping logic to prevent overshooting target.
        """
        if not hasattr(self, 'context'):
            return False

        requirements = self.context.get('requirements', {})
        attractions_needed = requirements.get('attractions_needed', 0)
        food_needed = requirements.get('food_places_needed', 0)
        total_needed = requirements.get('total_needed', 0)

        current_count = len(self.current_results)

        # Hard stop at 120% to prevent runaway (e.g., 45 → 54 max)
        if current_count >= total_needed * 1.2:
            logger.info(f"Hard stop: {current_count} places >= 120% of target ({total_needed * 1.2:.0f})")
            return True

        # If we're within 90-110% of target, check quality and balance
        if total_needed * 0.9 <= current_count <= total_needed * 1.1:
            # Count attractions vs food using _is_food_place helper
            food_count = sum(1 for p in self.current_results if self._is_food_place(p))
            attraction_count = current_count - food_count

            # Check if both requirements are reasonably met (within 80%)
            food_met = food_count >= food_needed * 0.8
            attraction_met = attraction_count >= attractions_needed * 0.8

            if food_met and attraction_met:
                logger.info(f"Requirements met: {attraction_count}/{attractions_needed} attractions, {food_count}/{food_needed} food")
                return True

        # Not yet at target
        return False

    def evaluate_results_quality(self, places: List[Dict], user_interests: List[str], user_uninterests: List[str], coordinates: List[Dict] = None) -> Dict:
        """
        Use LLM to evaluate search results quality.

        Args:
            places: List of found places
            user_interests: User's stated interests
            user_uninterests: User's stated uninterests/dislikes to avoid
            coordinates: Optional list of dicts with name, lat, lng, cluster for accurate geographic analysis

        Returns:
            Dict with quality scores and recommendations
        """
        if not places:
            return {
                "diversity_score": 0,
                "relevance_score": 0,
                "geographic_score": 0,
                "missing_categories": user_interests if user_interests else ["general attractions"],
                "recommendation": "expand_search"
            }

        # Prepare place summary for LLM
        place_summary = [
            f"{p.get('displayName', {}).get('text', p.get('name', 'Unknown'))} ({p.get('primaryType', 'unknown')})"
            for p in places[:20]  # Limit to first 20 for token efficiency
        ]

        # Prepare geographic data if coordinates provided
        geo_info = ""
        if coordinates:
            # Count places by cluster
            cluster_counts = {}
            for coord in coordinates[:20]:
                cluster = coord.get('cluster', 'unknown')
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

            geo_info = f"""

Actual Geographic Distribution (Singapore clusters):
{chr(10).join([f'  {cluster}: {count} places' for cluster, count in sorted(cluster_counts.items())])}

Sample locations (first 5):
{chr(10).join([f"  {coord.get('name', 'Unknown')}: cluster={coord.get('cluster', 'unknown')}, lat={coord.get('lat', 0):.4f}, lng={coord.get('lng', 0):.4f}" for coord in coordinates[:5]])}
"""

        uninterests_text = f"\nUser uninterests (AVOID): {', '.join(user_uninterests)}" if user_uninterests else ""

        prompt = f"""Evaluate these {len(places)} places for a Singapore trip.

User interests: {', '.join(user_interests) if user_interests else 'general sightseeing'}{uninterests_text}
Places found (sample): {', '.join(place_summary)}{geo_info}

Score on:
1. Diversity (0-10): Variety of experience types
2. Relevance (0-10): Match to user interests and AVOID uninterests
3. Geographic spread (0-10): Distribution across Singapore clusters (north, northeast, east, west, central, downtown, south)

Return ONLY valid JSON with no markdown:
{{
    "diversity_score": 0,
    "relevance_score": 0,
    "geographic_score": 0,
    "missing_categories": [],
    "recommendation": "expand_search" | "sufficient" | "refine_types"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"Quality evaluation: Diversity={result.get('diversity_score')}/10, "
                f"Relevance={result.get('relevance_score')}/10, "
                f"Geographic={result.get('geographic_score')}/10")
            return result

        except Exception as e:
            logger.error(f"Error in quality evaluation: {e}")
            return {
                "diversity_score": 5,
                "relevance_score": 5,
                "geographic_score": 5,
                "missing_categories": [],
                "recommendation": "continue"
            }

    def check_and_backfill(self, quality_result: Dict) -> List[Dict]:
        """
        Pattern 4: Analyze gaps and generate targeted backfill searches.

        Args:
            quality_result: Result from evaluate_results_quality()

        Returns:
            List of targeted search parameters to fill gaps
        """
        backfill_searches = []

        # Extract missing categories from quality evaluation
        missing_categories = quality_result.get('missing_categories', [])

        if not missing_categories:
            return backfill_searches

        logger.info(f"PATTERN 4 (Backfill): Identified {len(missing_categories)} missing categories")

        # Analyze current cluster distribution
        cluster_distribution = {}
        for place in self.current_results:
            cluster = place.get('location', {}).get('latitude', 0)  # Simplified
            cluster_id = 'unknown'
            # Determine cluster from coordinates
            if cluster:
                lat = place.get('location', {}).get('latitude')
                lng = place.get('location', {}).get('longitude')
                if lat and lng:
                    for cluster_name, bounds in SINGAPORE_GEOCLUSTERS_BOUNDARIES.items():
                        if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                            bounds['lon_min'] <= lng <= bounds['lon_max']):
                            cluster_id = cluster_name
                            break
            cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1

        # Identify underrepresented clusters
        all_clusters = list(SINGAPORE_GEOCLUSTERS_POINTS.keys())

        # Calculate target per cluster: aim for at least 10% of total places per cluster
        # For 45 places / 7 clusters = ~6 places per cluster minimum
        current_count = len(self.current_results)
        min_per_cluster = max(2, current_count // 10)  # At least 2, or 10% of total

        underrep_clusters = [c for c in all_clusters if cluster_distribution.get(c, 0) < min_per_cluster]

        logger.info(f"PATTERN 4: Current distribution: {cluster_distribution}")
        logger.info(f"PATTERN 4: Underrepresented clusters (< {min_per_cluster} places): {underrep_clusters}")

        # Map missing categories to place types
        for category in missing_categories[:3]:  # Limit to top 3
            # Use the map_interest_to_place_types tool to get valid Google types
            place_types = map_interest_to_place_types(category, self.client)

            if not place_types:
                place_types = ['tourist_attraction']  # Fallback

            # Create targeted searches for underrepresented clusters
            for cluster_name in underrep_clusters[:2]:  # Max 2 clusters
                cluster_center = SINGAPORE_GEOCLUSTERS_POINTS.get(cluster_name, {'lat': 1.3521, 'lon': 103.8198})

                search_params = {
                    'location': {
                        'lat': cluster_center['lat'],
                        'lng': cluster_center['lon']
                    },
                    'included_types': place_types,
                    'radius': 5000,  # Smaller radius for targeted search
                    'min_rating': 3.8,  # Slightly lower to get more options
                    'max_results': 10  # Limited results for gap filling
                }

                backfill_searches.append(search_params)
                logger.info(f"PATTERN 4: Backfill search for '{category}' in {cluster_name} cluster")

        return backfill_searches[:3]  # Max 3 backfill searches to avoid timeout

    def execute_react_search(self, max_iterations: int = 15) -> Dict:
        """
        Execute ReAct loop for adaptive place search.

        Returns:
            Dict with reasoning trace and found places
        """
        if not hasattr(self, 'context'):
            raise ValueError("Context not set. Call set_context() first.")

        duration_days = self.context.get('duration_days')
        requirements = self.context.get('requirements', {})
        user_interests = self.context.get('user_interests', [])
        user_uninterests = self.context.get('user_uninterests', [])
        excluded_types = self.context.get('excluded_types', [])
        accommodation = self.context.get('accommodation', {})

        # Initialize ReAct prompt
        initial_prompt = f"""Find ATTRACTION places for a {duration_days}-day Singapore trip.

REQUIREMENTS:
- Attractions needed: {requirements['attractions_needed']}
- User interests: {', '.join(user_interests) if user_interests else 'general sightseeing'}
- User uninterests/Excluded types: {', '.join(user_uninterests) if user_uninterests else 'none'}
- Accommodation: {accommodation.get('name', 'Not specified')} at ({accommodation.get('lat')}, {accommodation.get('lng')})

STRATEGY:
1. Search ONLY for attractions (museums, parks, temples, etc.) - NOT food places
2. Start with user interests if specified, AVOID uninterests
3. Use progressive relaxation: exact match → related categories → general attractions
4. Evaluate quality after each search batch
5. Expand radius or lower ratings if needed
6. Stop when attraction requirements met OR timeout (13 minutes)

Begin searching now."""

        self.messages.append({"role": "user", "content": initial_prompt})

        iteration = 0
        reasoning_trace = []

        while iteration < max_iterations and not self.requirements_met() and not self.check_timeout():
            iteration += 1
            logger.info(f"\n=== ReAct Iteration {iteration}/{max_iterations} ===")

            # Get LLM response with tool calling
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=self.messages,
                    tools=self.available_tools,
                    tool_choice="auto"
                )

                message = response.choices[0].message

                # Log thought/reasoning
                if message.content:
                    logger.info(f"THOUGHT: {message.content}")
                    reasoning_trace.append({"iteration": iteration, "thought": message.content})

                # Handle tool calls
                if message.tool_calls:
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        # FIX #4: Enhanced ACTION logging with arguments
                        args_summary = json.dumps({
                            k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 100 else f"{type(v).__name__}[{len(v) if isinstance(v, (list, dict)) else '...'}]")
                            for k, v in tool_args.items()
                        }, indent=2)
                        logger.info(f"ACTION: {tool_name}({args_summary})")

                        # Log includedTypes specifically for search_places to see what's being searched
                        if tool_name == "search_places" and "included_types" in tool_args:
                            included_types = tool_args["included_types"]
                            if isinstance(included_types, list):
                                logger.info(f"  -> includedTypes: {included_types}")
                            else:
                                logger.info(f"  -> includedTypes: [{included_types}]")

                        tool_result = self._execute_tool_call(tool_call)

                        # FIX #4: Enhanced OBSERVATION logging with detailed results
                        if isinstance(tool_result, list):
                            # Deduplicate by place_id
                            existing_ids = {p.get('id') or p.get('place_id') for p in self.current_results}
                            new_results = [p for p in tool_result if (p.get('id') or p.get('place_id')) not in existing_ids]
                            self.current_results.extend(new_results)

                            observation = f"Found {len(new_results)} new places (Total: {len(self.current_results)}/{self.context.get('requirements', {}).get('attractions_needed', 0) + self.context.get('requirements', {}).get('food_places_needed', 0)})"
                            logger.info(f"OBSERVATION: {observation}")
                        elif isinstance(tool_result, dict) and 'diversity_score' in tool_result:
                            # Quality evaluation result
                            observation = f"Quality scores - Diversity: {tool_result.get('diversity_score', 'N/A')}/10, Relevance: {tool_result.get('relevance_score', 'N/A')}/10, Geographic: {tool_result.get('geographic_score', 'N/A')}/10"
                            logger.info(f"OBSERVATION: {observation}")
                            if tool_result.get('missing_categories'):
                                logger.info(f"  Missing categories: {', '.join(tool_result['missing_categories'][:5])}")

                            # PATTERN 4: Check if backfill is needed
                            if tool_result.get('missing_categories') and not self.requirements_met():
                                logger.info(f"THOUGHT: Gaps detected. Initiating Pattern 4 backfill...")
                                backfill_searches = self.check_and_backfill(tool_result)

                                # Execute backfill searches immediately
                                for search_params in backfill_searches:
                                    try:
                                        included_types = search_params.get('included_types')
                                        logger.info(f"ACTION (Backfill): search_places")
                                        logger.info(f"  -> includedTypes: {included_types}")
                                        backfill_results = search_places(**search_params)

                                        if backfill_results:
                                            # Deduplicate
                                            existing_ids = set(p.get('id') or p.get('place_id') for p in self.current_results)
                                            new_backfill = [p for p in backfill_results if (p.get('id') or p.get('place_id')) not in existing_ids]

                                            if new_backfill:
                                                self.current_results.extend(new_backfill)
                                                logger.info(f"OBSERVATION (Backfill): Added {len(new_backfill)} places (Total: {len(self.current_results)})")
                                    except Exception as e:
                                        logger.error(f"Backfill search failed: {e}")
                        else:
                            observation = str(tool_result)[:200]
                            logger.info(f"OBSERVATION: {observation}")

                        # Add tool result to messages
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": observation if isinstance(observation, str) else json.dumps(observation)
                        })

                        reasoning_trace.append({
                            "iteration": iteration,
                            "action": tool_name,
                            "observation": observation
                        })
                else:
                    # No tool calls, LLM thinks we're done
                    logger.info("LLM indicated completion (no tool calls)")
                    break

            except Exception as e:
                logger.error(f"Error in ReAct iteration {iteration}: {e}")
                break

        # Final status
        logger.info(f"\n=== ReAct Search Complete ===")
        logger.info(f"Iterations: {iteration}")
        logger.info(f"Places found: {len(self.current_results)}")
        logger.info(f"Requirements met: {self.requirements_met()}")

        return {
            "reasoning_trace": reasoning_trace,
            "places_found": self.current_results,
            "iterations": iteration,
            "requirements_met": self.requirements_met()
        }

    def calculate_required_places(self, pace: str, duration_days: int, attraction_multiplier: float = None, food_multiplier: float = None) -> dict:
        """
        Calculate exact number of places needed.

        Formula:
        - Attractions: pace_value * days * attraction_multiplier (minimum = 2.0)
        - Food: geo_clusters * days * food_multiplier (minimum = 1.0)

        Args:
            pace: Trip pace (slow, relaxed, moderate, active, fast)
            duration_days: Number of days
            attraction_multiplier: Optional override (default from config: ATTRACTION_MULTIPLIER)
            food_multiplier: Optional override (default from config: FOOD_MULTIPLIER)

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

        # Get attraction multiplier from config or parameter (minimum 2.0)
        if attraction_multiplier is None:
            attraction_multiplier = ATTRACTION_MULTIPLIER

        # Get food multiplier from config or parameter (minimum 1.0)
        if food_multiplier is None:
            food_multiplier = FOOD_MULTIPLIER

        # Attractions calculation: pace * days * attraction_multiplier
        attractions_needed = int(pace_value * duration_days * attraction_multiplier)

        # Food calculation: num_clusters * days * food_multiplier
        # Example: 5 clusters * 3 days * 1.0 = 15 food places (1 from each cluster per day)
        num_clusters = len(SINGAPORE_GEOCLUSTERS_POINTS)
        food_places_needed = int(num_clusters * duration_days * food_multiplier)

        requirements = {
            "attractions_needed": attractions_needed,
            "food_places_needed": food_places_needed,
            "total_needed": attractions_needed + food_places_needed,
            "pace_value": pace_value,
            "attraction_multiplier": attraction_multiplier,
            "food_multiplier": food_multiplier,
            "num_geo_clusters": num_clusters,
        }

        logger.info(f"Requirements: {requirements['attractions_needed']} attractions + {requirements['food_places_needed']} food = {requirements['total_needed']} total")
        return requirements

    def map_interests(self, user_interests: List[str]) -> List[str]:
        """
        Map user interests to Google Places types using INTEREST_MAPPINGS.
        """
        all_place_types = []

        for interest in user_interests:
            interest_lower = interest.lower().strip()

            # Check direct mapping
            if interest_lower in INTEREST_MAPPINGS:
                place_types = INTEREST_MAPPINGS[interest_lower].copy()
                all_place_types.extend(place_types)
                continue

            # Check special categories
            if interest_lower in SPECIAL_INTEREST_CATEGORIES:
                place_types = SPECIAL_INTEREST_CATEGORIES[interest_lower]["include"].copy()
                all_place_types.extend(place_types)
                continue

            # Check for partial matches
            found = False
            for key, value in INTEREST_MAPPINGS.items():
                if key in interest_lower or interest_lower in key:
                    place_types = value.copy() if isinstance(value, list) else [value]
                    all_place_types.extend(place_types)
                    found = True
                    break

            if not found:
                logger.warning(f"No mapping found for interest: '{interest}'")

        # Deduplicate while preserving order
        seen = set()
        mapped = []
        for place_type in all_place_types:
            if place_type not in seen:
                seen.add(place_type)
                mapped.append(place_type)

        # Default to tourist_attraction if nothing mapped
        if not mapped:
            mapped.append("tourist_attraction")
            logger.info("No interests mapped, defaulting to 'tourist_attraction'")

        logger.info(f"Mapped interests: {mapped}")
        return mapped

    def get_auto_exclusions(self, user_interests: List[str]) -> List[str]:
        """
        Get place types that should be automatically excluded based on user interests.
        Prevents irrelevant results (e.g., pet stores when searching for aquariums).

        Args:
            user_interests: User's stated interests

        Returns:
            List of place types to exclude
        """
        auto_exclusions = []

        for interest in user_interests:
            interest_lower = interest.lower().strip()

            # Check if this interest has auto-exclusions
            for key, exclusions in INTEREST_AUTO_EXCLUSIONS.items():
                if key in interest_lower or interest_lower in key:
                    auto_exclusions.extend(exclusions)

        # Deduplicate
        auto_exclusions = list(set(auto_exclusions))

        if auto_exclusions:
            logger.info(f"Auto-exclusions for interests: {auto_exclusions}")

        return auto_exclusions

    def create_food_search_terms(self, food_interests: List[str], dietary_restrictions: List[str], days: int) -> List[str]:
        """
        Generate Google Places API food types based on food interests.
        Returns actual place types (not search strings).

        Note: DIETARY_EXCLUSIONS are handled separately in food_excluded_types,
        so this function only returns types to INCLUDE.

        Args:
            food_interests: User's food-related interests
            dietary_restrictions: User's dietary restrictions (used for logging only)
            days: Trip duration (unused, kept for compatibility)

        Returns:
            List of Google Places API food type strings
        """
        food_types = []

        # Check if user has specific food type interests
        has_restaurant = any('restaurant' in str(i).lower() for i in food_interests)
        has_cafe = any('cafe' in str(i).lower() or 'coffee' in str(i).lower() for i in food_interests)
        has_hawker = any('hawker' in str(i).lower() or 'food court' in str(i).lower() for i in food_interests)
        has_bakery = any('bakery' in str(i).lower() or 'pastry' in str(i).lower() for i in food_interests)

        # Build Google Places API food types based on interests
        if has_restaurant:
            # Prioritize restaurants
            food_types.extend(["restaurant", "fine_dining_restaurant"])

        if has_cafe:
            # Add cafes
            food_types.extend(["cafe", "coffee_shop"])

        if has_hawker:
            # Add hawker/food courts
            food_types.extend(["food_court"])

        if has_bakery:
            # Add bakeries
            food_types.extend(["bakery", "dessert_shop"])

        # If no specific interests, use general food types
        if not food_types:
            food_types = ["restaurant", "cafe", "food_court"]

        # Deduplicate while preserving order
        seen = set()
        unique_types = []
        for ftype in food_types:
            if ftype not in seen:
                seen.add(ftype)
                unique_types.append(ftype)

        # Return up to 3 types (Google Places API limit for includedTypes is 5, but we batch with others)
        return unique_types[:3]

    def search_food_by_geo_clusters(
        self,
        food_search_terms: List[str],
        duration_days: int,
        food_multiplier: float,
        excluded_types: List[str] = None,
        destination_city: str = "Singapore"
    ) -> List[Dict]:
        """
        Search for food places distributed across Singapore geo clusters.

        For each cluster:
        - Target: duration_days * food_multiplier food places
        - Start: radius=5000m, min_rating=4.5
        - Fallback: expand radius and lower rating (handled by search_with_requirements)

        Args:
            food_search_terms: List of food search terms/types
            duration_days: Number of trip days
            food_multiplier: Multiplier from config (default 1.0)
            excluded_types: Types to exclude
            destination_city: City filter (default "Singapore")

        Returns:
            List of food places distributed across all clusters
        """
        # Calculate target per cluster
        target_per_cluster = int(duration_days * food_multiplier)
        total_clusters = len(SINGAPORE_GEOCLUSTERS_POINTS)

        print(f"Target: {target_per_cluster} food place(s) from each of {total_clusters} clusters")
        print(f"Total target: {target_per_cluster * total_clusters} food places\n")

        all_food_results = []
        cluster_stats = {}

        for cluster_name, cluster_loc in SINGAPORE_GEOCLUSTERS_POINTS.items():
            # Get cluster-specific search parameters or use defaults
            cluster_params = FOOD_SEARCH_PARAMS_BY_CLUSTER.get(cluster_name, {})
            min_rating = cluster_params.get("min_rating", 4.5)  # Default 4.5
            initial_radius = cluster_params.get("initial_radius", 5000)  # Default 5000m

            print(f"\n--- Cluster: {cluster_name.upper()} ({cluster_loc['lat']:.4f}, {cluster_loc['lon']:.4f}) ---")
            print(f"    Search params: min_rating={min_rating}, initial_radius={initial_radius}m")

            cluster_results = []

            # Search each food term in this cluster
            for food_term in food_search_terms:
                if len(cluster_results) >= target_per_cluster:
                    print(f"[OK] Cluster target reached: {len(cluster_results)}/{target_per_cluster}")
                    break

                # Convert lon to lng for consistency
                location = {"lat": cluster_loc["lat"], "lng": cluster_loc["lon"]}

                # Use search_with_requirements with cluster-specific parameters
                # Fallback logic (expand radius, lower rating) is built-in
                results = self.search_with_requirements(
                    location=location,
                    included_types=food_term,
                    min_rating=min_rating,  # Cluster-specific rating
                    max_results_needed=target_per_cluster - len(cluster_results),
                    search_type='food',
                    excluded_types=excluded_types,
                    destination_city=destination_city,
                    initial_radius=initial_radius  # Cluster-specific radius
                )

                # Deduplicate within cluster
                existing_ids = {p.get('id') or p.get('place_id') for p in cluster_results}
                new_results = [p for p in results if (p.get('id') or p.get('place_id')) not in existing_ids]

                cluster_results.extend(new_results)
                print(f"  {food_term}: +{len(new_results)} (cluster total: {len(cluster_results)}/{target_per_cluster})")

            # Track cluster stats
            cluster_stats[cluster_name] = len(cluster_results)
            all_food_results.extend(cluster_results)

            print(f"[OK] {cluster_name}: {len(cluster_results)}/{target_per_cluster} food places")

        # Print summary
        print(f"\n{'='*60}")
        print(f"FOOD SEARCH SUMMARY (Geo-Cluster Distribution)")
        print(f"{'='*60}")
        for cluster_name, count in cluster_stats.items():
            status = "[OK]" if count >= target_per_cluster else "[WARN]"
            print(f"{status} {cluster_name:12} {count:2}/{target_per_cluster} places")
        print(f"{'='*60}")
        print(f"Total food places: {len(all_food_results)}/{target_per_cluster * total_clusters}")
        print(f"{'='*60}\n")

        return all_food_results

    def search_with_requirements(self, location, included_types, min_rating, max_results_needed, search_type='attraction', excluded_types=None, destination_city=None, initial_radius=None):
        """
        Deterministic progressive search with radius expansion and rating relaxation.

        Used by search_food_by_geo_clusters() for predictable, calculation-based food search.
        For attractions, use execute_react_search() which adapts intelligently.

        Progressive search: Expand radius FIRST, then relax rating ONLY if needed

        Strategy progressively:
        1. Widens search radius (initial_radius → ... → 35km)
        2. Relaxes rating requirements (start high, gradually lower by ~0.1-0.2 per level)

        The entire rating curve shifts based on the passed min_rating:
        - Default attractions: starts at 4.0, relaxes down to 2.5 (never below 3.0)
        - Default food: starts at 4.5, relaxes down to 3.5 (stricter quality control)
        - Custom (e.g., shopping malls at 3.5): shifts entire curve down by 0.5

        Args:
            location: Location dict with lat/lng
            included_types: Place type(s) to search for (string or list)
            min_rating: Starting minimum rating
            max_results_needed: Target number of results
            search_type: 'attraction' or 'food' (determines rating floor)
            excluded_types: Place types to exclude (optional)
            destination_city: City name to filter results by (e.g., "Singapore")
            initial_radius: Starting search radius in meters (default: 10000m)

        Returns:
            List of places (up to max_results_needed)
        """
        # Progressive search strategy: For each rating level, sweep through all radii
        # Example: 10km@4.0, 20km@4.0, ..., 35km@4.0, then 10km@3.9, 20km@3.9, ..., 35km@3.9, ...

        strategies = []
        max_radius = 35000  # Google Places API maximum
        radius_step = 10000  # Expand by 10km each time (reduced API calls)
        start_radius = initial_radius if initial_radius else radius_step  # Use custom initial radius if provided

        # Food: stricter floor (3.5), Attractions: more relaxed floor (3.0)
        rating_floor = 3.5 if search_type == 'food' else 3.0
        rating_decrement = 0.1

        # For each rating level (starting at min_rating, decreasing by 0.1 until floor)
        current_rating = min_rating
        while current_rating >= rating_floor:
            # Sweep through all radii at this rating level (start_radius, start_radius+10km, ..., max_radius)
            for radius in range(start_radius, max_radius + 1, radius_step):
                strategies.append({
                    'radius': radius,
                    'max_results': 20,
                    'min_rating': round(current_rating, 1)
                })
            current_rating -= rating_decrement

        logger.info(f"Generated {len(strategies)} search strategies (ratings: {min_rating} down to {rating_floor}, radii: {start_radius/1000}km to {max_radius/1000}km per rating)")

        all_results = []

        for level, strategy in enumerate(strategies, 1):
            if len(all_results) >= max_results_needed:
                print(f"Target reached: {len(all_results)}/{max_results_needed} for '{included_types}'")
                break

            radius = strategy['radius']
            max_results = strategy['max_results']
            current_min_rating = strategy['min_rating']

            still_needed = max_results_needed - len(all_results)
            logger.info(f"[Level {level}/{len(strategies)}] Searching '{included_types}' at radius={radius}m, max_results={max_results}, rating>={current_min_rating} (need {still_needed} more)")
            print(f"  [{level}/{len(strategies)}] {included_types}: {radius/1000}km, max={max_results}, rating>={current_min_rating:.1f}...")

            results = search_places(
                location=location,
                included_types=included_types,
                excluded_types=excluded_types,
                radius=radius,
                max_results=max_results,
                min_rating=current_min_rating,
                destination_city=destination_city
            )

            # Deduplicate by place_id (API v1 uses 'id' field)
            existing_ids = {p.get('id') or p.get('place_id') for p in all_results}
            new_results = [p for p in results if (p.get('id') or p.get('place_id')) not in existing_ids]

            all_results.extend(new_results)
            logger.info(f"Found {len(new_results)} new results at level {level} (total: {len(all_results)})")
            print(f"      -> +{len(new_results)} new (total: {len(all_results)}/{max_results_needed})")

            # If we got enough, stop expanding
            if len(all_results) >= max_results_needed:
                print(f"Target reached for '{included_types}'")
                break

            # If no new results at current level, continue to next level
            if not new_results:
                print(f"      WARNING: No new results, trying next level...")
                continue

        # Return up to max_results_needed
        final_count = min(len(all_results), max_results_needed)
        if final_count < max_results_needed:
            print(f"WARNING: Only found {final_count}/{max_results_needed} for '{included_types}' (type: {search_type})")

        return all_results[:max_results_needed]

    # === FORMATTING METHODS ===

    def format_results(self, raw_places: List[Dict], enrichments: Dict = None) -> List[Dict]:
        """
        Format raw places from Google Places API v1 into structured schema.

        Args:
            raw_places: List of raw place data from search API (API v1 format)
            enrichments: Optional dict with 'details' data from get_place_details()

        Returns:
            List of formatted place dictionaries
        """
        if not enrichments:
            enrichments = {'details': {}}

        details_by_id = enrichments.get('details', {})

        formatted_places = []
        for place in raw_places:
            # Extract place_id from API v1 format (id field: "places/{id}")
            place_id = place.get('id', '')

            # Ensure place_id has "places/" prefix for lookup
            place_id_with_prefix = place_id if place_id.startswith('places/') else f"places/{place_id}"

            # Get enrichment data for this place (details dict uses full "places/{id}" format)
            details_data = details_by_id.get(place_id_with_prefix)

            # Format the place with enriched data
            formatted_place = self._format_single_place(place, details_data)
            formatted_places.append(formatted_place)

        return formatted_places

    def _format_single_place(self, place_data: Dict, details_data: Dict = None) -> Dict:
        """
        Format single place from raw Google Places API v1 data to structured schema.

        Args:
            place_data: Raw place data from search API (API v1 format)
            details_data: Optional detailed data from get_place_details (API v1 format)
        """
        # Merge details if available
        if details_data:
            place_data = {**place_data, **details_data}

        # Extract place_id (API v1: id field, format: "places/{place_id}")
        place_id = place_data.get('id', '')
        if place_id.startswith('places/'):
            place_id = place_id.replace('places/', '')

        # Extract name (API v1: displayName.text)
        display_name = place_data.get('displayName', {})
        name = display_name.get('text', 'Unknown Place') if isinstance(display_name, dict) else display_name

        # Extract address (API v1: formattedAddress)
        address = place_data.get('formattedAddress')

        # Extract geo coordinates (API v1: location.latitude/longitude)
        geo = self._extract_geo(place_data)
        geo_cluster_id = None
        if geo and geo.get('latitude') and geo.get('longitude'):
            geo_cluster_id = calculate_geo_cluster(geo['latitude'], geo['longitude'])

        # Extract types and primaryType (API v1: types[], primaryType)
        all_types = place_data.get('types', [])
        primary_type_v1 = place_data.get('primaryType')
        if primary_type_v1 and primary_type_v1 not in all_types:
            all_types.insert(0, primary_type_v1)

        # Extract rating and review count (API v1: rating, userRatingCount)
        rating = place_data.get('rating')
        reviews_count = place_data.get('userRatingCount')

        # Extract priceLevel (API v1: enum string -> int)
        price_level = self._map_price_level_from_v1(place_data.get('priceLevel'))

        # Extract opening hours (API v1: regularOpeningHours.weekdayDescriptions)
        opening_hours = self._extract_opening_hours_v1(place_data)

        # Extract accessibility (API v1: accessibilityOptions.wheelchairAccessibleEntrance)
        accessibility_options = self._extract_accessibility_v1(place_data)

        # Extract description (API v1: editorialSummary.text)
        editorial = place_data.get('editorialSummary', {})
        editorial_text = editorial.get('text') if isinstance(editorial, dict) else editorial

        # If no editorialSummary from API, generate description using LLM
        if not editorial_text and self.client:
            description_data = {
                'name': name,
                'address': address or 'Singapore',
                'latitude': geo.get('latitude', 0) if geo else 0,
                'longitude': geo.get('longitude', 0) if geo else 0,
                'neighborhood': address.split(',')[-2].strip() if address and ',' in address else 'Singapore',
                'type': primary_type_v1 or (all_types[0] if all_types else 'tourist_attraction')
            }
            description = generate_place_description(description_data, openai_client=self.client)
        else:
            description = editorial_text or f"A {primary_type_v1 or (all_types[0] if all_types else 'place')} in Singapore"

        # Extract website (API v1: websiteUri)
        website = place_data.get('websiteUri')

        # Generate enhanced tags
        tags = generate_tags(
            place_type=primary_type_v1 or (all_types[0] if all_types else 'tourist_attraction'),
            all_types=all_types,
            accessibility_options=accessibility_options,
            price_level=price_level,
            rating=rating,
            description=description,
            name=name,
            reviews_count=reviews_count,
            openai_client=self.client
        )

        # Calculate onsite carbon emissions
        # Determine which type to use for carbon calculation: primaryType first, fallback to first type in types
        carbon_type = primary_type_v1
        if not carbon_type and all_types:
            # Fallback: use first type from types array (no LLM)
            carbon_type = all_types[0]

        onsite_co2_kg = None
        low_carbon_score = None
        if carbon_type:
            try:
                details = get_place_carbon_details(carbon_type, num_people=1)
                onsite_co2_kg = details.get("co2e_total_kg")
                low_carbon_score = details.get("low_carbon_score")
            except Exception as e:
                logger.warning(f"Failed to calculate carbon for {carbon_type}: {e}")

        return {
            "place_id": place_id,
            "name": remove_unicode(name),
            "types": place_data.get('types'),
            "primaryType": place_data.get('primaryType', 'point_of_interest'),
            "cost_sgd": self._map_price_level_to_cost(price_level),
            "price_level": price_level,
            "onsite_co2_kg": onsite_co2_kg,
            "low_carbon_score": low_carbon_score,
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
            "description": remove_unicode(description),
            "links": place_data.get('website'),
            "rating": place_data.get('rating'),
            "userRatingCount": place_data.get('userRatingCount'),  # Number of user ratings from Google
            "tags": tags
        }

    def _extract_geo(self, place_data: Dict) -> Optional[Dict]:
        """
        Extract geo coordinates from place data.
        Supports both API v1 (location.latitude/longitude) and legacy format (geometry.location.lat/lng).
        """
        # Try API v1 format first (location.latitude/longitude)
        location = place_data.get('location')
        if location and isinstance(location, dict):
            lat = location.get('latitude')
            lng = location.get('longitude')
            if lat is not None and lng is not None:
                return {"latitude": lat, "longitude": lng}

        # Fallback to legacy format (geometry.location.lat/lng)
        geometry = place_data.get('geometry')
        if geometry:
            location = geometry.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            if lat is not None and lng is not None:
                return {"latitude": lat, "longitude": lng}

        return None

    def _map_price_level_from_v1(self, price_level_str: Optional[str]) -> Optional[int]:
        """
        Map Google Places API v1 priceLevel enum to integer (0-4).

        API v1 enum:
        - PRICE_LEVEL_FREE -> 0
        - PRICE_LEVEL_INEXPENSIVE -> 1
        - PRICE_LEVEL_MODERATE -> 2
        - PRICE_LEVEL_EXPENSIVE -> 3
        - PRICE_LEVEL_VERY_EXPENSIVE -> 4
        """
        if not price_level_str:
            return None

        price_level_map = {
            'PRICE_LEVEL_FREE': 0,
            'PRICE_LEVEL_INEXPENSIVE': 1,
            'PRICE_LEVEL_MODERATE': 2,
            'PRICE_LEVEL_EXPENSIVE': 3,
            'PRICE_LEVEL_VERY_EXPENSIVE': 4
        }
        return price_level_map.get(price_level_str)

    def _extract_opening_hours_v1(self, place_data: Dict) -> Dict:
        """
        Extract and parse opening hours from API v1 format.
        API v1: regularOpeningHours.weekdayDescriptions
        """
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

        regular_hours = place_data.get('regularOpeningHours', {})
        weekday_descriptions = regular_hours.get('weekdayDescriptions', [])

        if weekday_descriptions:
            opening_hours = self._parse_opening_hours({'weekday_text': weekday_descriptions})

        return opening_hours

    def _extract_accessibility_v1(self, place_data: Dict) -> List[str]:
        """
        Extract accessibility options from API v1 format.
        API v1: accessibilityOptions.wheelchairAccessibleEntrance
        """
        accessibility_options = []
        accessibility = place_data.get('accessibilityOptions', {})

        if accessibility.get('wheelchairAccessibleEntrance'):
            accessibility_options.append("wheelchair_accessible_entrance")

        return accessibility_options

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

def research_places(input_file: str, output_file: str = None, session_id: str = None) -> dict:
    """
    Research and format Singapore places using ReAct pattern with deterministic food search.

    Strategy:
    - Attractions: ReAct intelligent search (adaptive, LLM-driven)
    - Food: Deterministic geo-cluster search (7 clusters x days x 1.0)

    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results (default: ResearchAgent/output.json)
        session_id: Session identifier for retrieval_id generation

    Returns:
        Dictionary with formatted places

    Example:
        results = research_places('inputs/input.json')
    """
    # Load input
    input_data = load_input_file(input_file)
    if not input_data:
        return {"error": "Failed to load input file"}

    # Extract parameters
    requirements_data = input_data.get('requirements', {})
    pace = requirements_data.get('pace', 'moderate')
    duration_days = requirements_data.get('duration_days', 1)
    location = requirements_data.get('optional', {}).get('accommodation_location', {})

    # Get travelers
    travelers = requirements_data.get('travelers', {})
    num_travelers = (travelers.get('adults', 0) + travelers.get('children', 0)) or 1

    # Initialize agent
    agent = PlacesResearchAgent(num_travelers=num_travelers)
    start_time = time.time()

    # Get interests and exclusions
    user_interests = requirements_data.get('optional', {}).get('interests', [])
    if isinstance(user_interests, str):
        user_interests = [user_interests]

    user_uninterests = requirements_data.get('optional', {}).get('uninterests', [])
    if isinstance(user_uninterests, str):
        user_uninterests = [user_uninterests]

    # Calculate requirements
    requirements = agent.calculate_required_places(pace, duration_days)

    # Get auto-exclusions based on interests (e.g., exclude pet_store when searching for aquarium)
    auto_exclusions = agent.get_auto_exclusions(user_interests)

    # Set context and execute ReAct
    agent.set_context({
        'duration_days': duration_days,
        'requirements': requirements,
        'user_interests': user_interests,
        'user_uninterests': user_uninterests,
        'excluded_types': auto_exclusions,  # Use auto-exclusions
        'accommodation': location,
        'start_time': start_time,
    })

    logger.info("\\n=== Starting ReAct Search (Attractions Only) ===")
    react_result = agent.execute_react_search(max_iterations=15)

    attraction_results = react_result['places_found']
    logger.info(f"ReAct found {len(attraction_results)} attraction place(s)")

    # ===================================================================
    # DETERMINISTIC FOOD SEARCH (Geo-Cluster Based)
    # ===================================================================
    logger.info("\\n=== Starting Deterministic Food Search ===")

    # Extract food-related interests and dietary restrictions
    dietary_restrictions = requirements_data.get('optional', {}).get('dietary_restrictions', [])
    if isinstance(dietary_restrictions, str):
        dietary_restrictions = [dietary_restrictions]

    user_uninterests = requirements_data.get('optional', {}).get('uninterests', [])
    if isinstance(user_uninterests, str):
        user_uninterests = [user_uninterests]

    # Extract food interests from user interests
    food_interests = [i for i in user_interests if any(
        food_kw in str(i).lower() for food_kw in ['restaurant', 'cafe', 'food', 'dining', 'hawker', 'bar', 'bakery']
    )]

    # Generate food search terms
    food_search_terms = agent.create_food_search_terms(food_interests, dietary_restrictions, duration_days)
    logger.info(f"Food interests: {food_interests}")
    logger.info(f"Food search terms: {food_search_terms}")

    # Get food exclusions from dietary restrictions and uninterests
    food_excluded_types = []
    for restriction in dietary_restrictions:
        food_excluded_types.extend(DIETARY_EXCLUSIONS.get(restriction.lower(), []))

    # Add food-related uninterests to exclusions
    food_uninterests = [u for u in user_uninterests if any(
        food_kw in str(u).lower() for food_kw in ['restaurant', 'cafe', 'food', 'dining', 'hawker', 'bar', 'bakery']
    )]
    food_excluded_types.extend(food_uninterests)

    # Run deterministic geo-cluster food search
    food_results = agent.search_food_by_geo_clusters(
        food_search_terms=food_search_terms,
        duration_days=duration_days,
        food_multiplier=requirements['food_multiplier'],
        excluded_types=food_excluded_types if food_excluded_types else None,
        destination_city=requirements_data.get('destination_city', 'Singapore')
    )

    # Deduplicate: remove food places that are already in attraction results
    existing_attraction_ids = {(p.get('id') or p.get('place_id')) for p in attraction_results}
    unique_food_results = [p for p in food_results if (p.get('id') or p.get('place_id')) not in existing_attraction_ids]

    duplicates_found = len(food_results) - len(unique_food_results)
    if duplicates_found > 0:
        logger.info(f"Removed {duplicates_found} duplicate(s) (food places already in attractions)")

    # Combine attractions + unique food places
    all_places = attraction_results + unique_food_results
    logger.info(f"\\n=== Total places collected: {len(all_places)} (attractions: {len(attraction_results)}, unique food: {len(unique_food_results)}) ===")

    # FIX #1 & #2: Format places with full enrichment (tags, descriptions, carbon, geo_cluster_id)
    # The format_results method already calls _format_single_place which handles all enrichment
    logger.info("\\n=== Formatting and Enrichment Phase ===")
    formatted_places = agent.format_results(all_places)

    # Log enrichment stats
    cluster_distribution = {}
    enriched_count = sum(1 for p in formatted_places if p.get('tags') and p.get('onsite_co2_kg') and p.get('geo_cluster_id'))
    for p in formatted_places:
        cid = p.get('geo_cluster_id', 'unknown')
        cluster_distribution[cid] = cluster_distribution.get(cid, 0) + 1

    logger.info(f"Formatted {len(formatted_places)} places with full enrichment")
    logger.info(f"Enriched: {enriched_count}/{len(formatted_places)} places have tags, carbon scores, and geo clusters")
    logger.info(f"Cluster distribution: {cluster_distribution}")

    elapsed = time.time() - start_time

    # Separate attractions from food places for detailed counts
    attractions = [p for p in formatted_places if not agent._is_food_place(p)]
    food_places = [p for p in formatted_places if agent._is_food_place(p)]

    # Calculate cluster distribution for attractions
    attraction_clusters = {}
    for place in attractions:
        cluster = place.get('geo_cluster_id', 'unknown')
        attraction_clusters[cluster] = attraction_clusters.get(cluster, 0) + 1

    # Calculate cluster distribution for food
    food_clusters = {}
    for place in food_places:
        cluster = place.get('geo_cluster_id', 'unknown')
        food_clusters[cluster] = food_clusters.get(cluster, 0) + 1

    # Build structured count objects with cluster breakdown
    attraction_count_obj = {
        "total": len(attractions),
        "north": attraction_clusters.get('north', 0),
        "northeast": attraction_clusters.get('northeast', 0),
        "east": attraction_clusters.get('east', 0),
        "west": attraction_clusters.get('west', 0),
        "south": attraction_clusters.get('south', 0),
        "central": attraction_clusters.get('central', 0),
        "downtown": attraction_clusters.get('downtown', 0)
    }

    food_count_obj = {
        "total": len(food_places),
        "north": food_clusters.get('north', 0),
        "northeast": food_clusters.get('northeast', 0),
        "east": food_clusters.get('east', 0),
        "west": food_clusters.get('west', 0),
        "south": food_clusters.get('south', 0),
        "central": food_clusters.get('central', 0),
        "downtown": food_clusters.get('downtown', 0)
    }

    # FIX #3: Add reasoning trace and requirements status to output
    result = {
        **input_data,
        "retrieval": {
            "retrieval_id": f"ret_{session_id or datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "time_elapsed": elapsed,
            "places_found": len(formatted_places),
            "attraction_count": attraction_count_obj,
            "food_count": food_count_obj,
            "conditions": {
                "attractions_needed": requirements['attractions_needed'],
                "food_places_needed": requirements['food_places_needed'],
                "total_needed": requirements['total_needed'],
                "pace_value": requirements['pace_value'],
                "attraction_multiplier": requirements['attraction_multiplier'],
                "food_multiplier": requirements['food_multiplier'],
                "num_geo_clusters": requirements['num_geo_clusters']
            },
            "react_iterations": react_result['iterations'],
            "react_requirements_met": react_result.get('requirements_met', False),
            "react_reasoning": react_result.get('reasoning_trace', []),
            "cluster_distribution": cluster_distribution
        },
        "places": formatted_places
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

    return result


# ## Main Execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        result = research_places(input_file, output_file)
        print(f"Found {result['retrieval']['places_found']} places")
    else:
        print("Usage: python main.py <input_file> [output_file]")
        print("Example: python main.py inputs/input.json outputs/output.json")