#!/usr/bin/env python3
"""
Singapore Sustainable Tourism Agent
Simplified agent focused on discovering eco-friendly, sustainable tourism options.
Uses the new SustainableTourismAgent for streamlined operation.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.singapore_places_api import SingaporePlacesAPI
# from src.llm_evaluator import LLMPlaceEvaluator  # Removed - merged into main agent
# from src.interest_mapper import InterestMapper  # Removed - merged into main agent
from src.transport_analyzer import SingaporeTransportAnalyzer
from src.carbon_calculator import SingaporeCarbonCalculator
from src.place_relationship_graph import PlaceGraphGenerator

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Configure logging for agent reasoning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_reasoning.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

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
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

class SingaporeAttractionAgent:
    """
    Agentic system for discovering Singapore attractions using tool-based reasoning.
    
    The agent uses RGC (Role & Result + Goal + Context & Constraint) prompting
    to make decisions about tool usage and iteratively refine results.
    """
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_llm_clients()
        
        # Available tools - agent will decide which ones to use
        self.available_tools = {
            'places_api': {
                'instance': SingaporePlacesAPI(),
                'description': 'Searches Singapore places using Google Places API with interest mapping done internally',
                'input': 'user_interests, accommodation_location, search_radius',
                'output': 'raw_places_data'
            },
            'graph_generator': {
                'instance': PlaceGraphGenerator(),
                'description': 'Generates place relationship graphs and clusters',
                'input': 'evaluated_places',
                'output': 'place_clusters'
            },
            'transport_analyzer': {
                'instance': SingaporeTransportAnalyzer(),
                'description': 'Analyzes transport options between clustered places',
                'input': 'place_clusters, accommodation_location',
                'output': 'transport_matrix'
            },
            'carbon_calculator': {
                'instance': SingaporeCarbonCalculator(),
                'description': 'Calculates carbon emissions for transport routes using Climatiq API',
                'input': 'transport_matrix',
                'output': 'carbon_scored_transport_matrix'
            }
        }
        
        # Agent state
        self.agent_state = {
            'current_results': [],
            'iteration_count': 0,
            'max_iterations': 8,
            'required_places': 0,
            'search_strategy': 'initial',
            'current_search_radius_km': 0,  # Will be set from user input
            'base_search_radius_km': 0,     # Original radius for expansion calculations
            'last_action': 'none',          # Track last tool used
            'last_result_summary': 'Starting workflow'  # Simple summary of last result
        }

        # Cache system for tool responses
        self.cache_file = 'cache.json'
        self.tool_cache = {
            'iteration_history': [],
            'accumulated_data': {}
        }
    
    def _initialize_llm_clients(self):
        """Initialize LLM clients for agent reasoning."""
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if openai_key and OpenAI:
            self.openai_client = OpenAI(api_key=openai_key)
            logger.info("Agent initialized with OpenAI client")
        
        if anthropic_key and Anthropic:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
            logger.info("Agent initialized with Anthropic client")
        
        if not self.openai_client and not self.anthropic_client:
            raise ValueError("Agent requires OPENAI_API_KEY or ANTHROPIC_API_KEY for reasoning")

    def save_tool_response_to_cache(self, tool_name: str, iteration: int, response_data: Any):
        """Save any tool response to cache.json for LLM context in next iteration."""
        try:
            import time
            cache_entry = {
                'iteration': iteration,
                'tool': tool_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'response': response_data
            }

            self.tool_cache['iteration_history'].append(cache_entry)

            # Save to file
            with open(self.cache_file, 'w') as f:
                json.dump(self.tool_cache, f, indent=2, default=str)

            logger.info(f"CACHE: Saved {tool_name} response from iteration {iteration}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_cache_context_for_llm(self) -> str:
        """Get all previous tool responses as context for LLM reasoning."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.tool_cache = cached_data

            if not self.tool_cache['iteration_history']:
                return "No previous tool responses available."

            # Build context from all previous tool responses
            context = "PREVIOUS TOOL RESPONSES:\n"

            for entry in self.tool_cache['iteration_history']:
                tool_name = entry['tool']
                iteration = entry['iteration']
                response = entry['response']

                # Truncate long responses for context
                response_preview = str(response)[:300] + "..." if len(str(response)) > 300 else str(response)

                context += f"\nIteration {iteration} - {tool_name}:\n{response_preview}\n"

            return context

        except Exception as e:
            logger.error(f"Failed to load cache context: {e}")
            return "Error loading previous tool responses."

    def map_interests_to_place_types(self, interests: List[str]) -> List[str]:
        """
        Map user interests to Google Places API types using predefined mappings.

        Args:
            interests: List of user interests (e.g., ["gardens", "cultural_shows"])

        Returns:
            List of Google Places API types (excluding food-related types)
        """
        try:
            if not interests:
                return ["tourist_attraction"]  # Default fallback

            logger.info(f"Mapping interests to place types: {interests}")

            # Direct mapping dictionary - no LLM needed for this simple lookup
            interest_mappings = {
                "gardens": ["park", "tourist_attraction"],
                "light_walks": ["park", "tourist_attraction"],
                "cultural_shows": ["museum", "art_gallery", "tourist_attraction"],
                "riverfronts": ["tourist_attraction", "park"],
                "heritage": ["museum", "tourist_attraction"],
                "shopping": ["shopping_mall", "store", "department_store"],
                "temples": ["church", "hindu_temple", "mosque", "synagogue"],
                "beaches": ["tourist_attraction", "park"],
                "viewpoints": ["tourist_attraction"],
                "nature": ["park", "zoo", "aquarium", "tourist_attraction"],
                "architecture": ["tourist_attraction", "museum"],
                "family": ["amusement_park", "zoo", "aquarium", "tourist_attraction"],
                "adventure": ["tourist_attraction", "amusement_park"],
                "nightlife": ["night_club", "casino"]
            }

            mapped_types = set()
            for interest in interests:
                interest_lower = interest.lower()
                if interest_lower in interest_mappings:
                    mapped_types.update(interest_mappings[interest_lower])
                else:
                    # Fallback for unknown interests
                    mapped_types.add("tourist_attraction")

            result = list(mapped_types) if mapped_types else ["tourist_attraction"]
            logger.info(f"Mapped interests to place types: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in interest mapping: {e}")
            return ["tourist_attraction"]  # Safe fallback
    
    def calculate_required_places(self, pace: str, duration_days: int) -> int:
        """
        Agent's internal goal-setting: Calculate required number of places.
        This sets the target for the agent to achieve through tool usage.
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
            
            # Cap at reasonable limits
            required_places = min(required_places, 60)
            required_places = max(required_places, 5)  # Minimum 5 places
            
            logger.info(f"Agent goal set: {required_places} places for {pace} pace over {duration_days} days")
            return required_places
            
        except (ValueError, TypeError, AttributeError):
            logger.warning("Failed to calculate required places, using default: 15")
            return 15

    def validate_output_quantity(self, results: List[Dict], required_count: int) -> Dict[str, Any]:
        """
        Agent's self-assessment: Validate if current results meet requirements.
        Used by agent to decide if more tool usage is needed.
        
        Note: Accommodation is excluded from attraction count as it's not an attraction.
        """
        # Filter out accommodation from attraction count
        attractions_only = [result for result in (results or []) 
        if result.get('place_id') != 'accommodation' 
        and not result.get('is_accommodation', False)]
        
        current_count = len(attractions_only)
        meets_requirement = current_count >= required_count
        
        validation = {
            'sufficient': meets_requirement,
            'current_count': current_count,
            'required_count': required_count,
            'deficit': max(0, required_count - current_count),
            'success_rate': min(1.0, current_count / required_count) if required_count > 0 else 0.0,
            'total_places_including_accommodation': len(results) if results else 0
        }
        
        logger.info(f"Agent self-assessment - Attractions: {current_count}, Required: {required_count}, Sufficient: {meets_requirement} (Total places including accommodation: {validation['total_places_including_accommodation']})")
        return validation
    
    def create_rgc_prompt(self, user_input: Dict[str, Any], current_state: Dict[str, Any]) -> str:
        """
        Create RGC (Role & Result + Goal + Context & Constraint) prompt for agent reasoning.
        
        This prompt helps the agent decide what tools to use and how to proceed.
        """
        
        # Get cached context from previous tool responses
        cache_context = self.get_cache_context_for_llm()

        prompt = f"""You are a Singapore Attraction Discovery Agent.

YOUR GOAL: Find {current_state.get('required_places', 0)} attractions for the user.

CURRENT SITUATION:
- You have found {len([r for r in current_state.get('current_results', []) if r.get('place_id') != 'accommodation'])} attractions so far
- You need {current_state.get('required_places', 0)} total attractions
- Status: {current_state.get('last_result_summary', 'Starting workflow')}

{cache_context}

INSTRUCTIONS:
Decide which tool to use next and respond with ONLY:

Action: tool_name

Available tools: places_api, graph_generator, transport_analyzer, carbon_calculator

CURRENT CONTEXT:
- Iteration: {current_state.get('iteration_count', 0)}/{current_state.get('max_iterations', 5)}
- Current attractions: {len([r for r in current_state.get('current_results', []) if r.get('place_id') != 'accommodation'])}/{current_state.get('required_places', 15)} found (accommodation does NOT count)
- Last action: {current_state.get('last_action', 'none')} - {current_state.get('last_result_summary', 'Starting workflow')}
- Status: {'SUFFICIENT - Ready for next phase' if len([r for r in current_state.get('current_results', []) if r.get('place_id') != 'accommodation']) >= current_state.get('required_places', 15) else f'NEED MORE - Deficit of {current_state.get("required_places", 15) - len([r for r in current_state.get("current_results", []) if r.get("place_id") != "accommodation"])} attractions'}
- Search radius: {current_state.get('current_search_radius_km', 25)}km
- User interests: {user_input.get('interests', 'general sightseeing')}
- User location: {user_input.get('accommodation_location', {}).get('lat', 1.3521)}, {user_input.get('accommodation_location', {}).get('lon', 103.8198)}

AVAILABLE TOOLS:
1. places_api: Maps user interests to place types internally, searches Singapore places using Google Places API, and agent evaluates them directly (input: user_interests, accommodation_location, search_radius → output: evaluated_places)
2. graph_generator: Generates place relationship graphs and clusters (input: evaluated_places → output: place_clusters)
3. transport_analyzer: Analyzes transport options between clustered places (input: place_clusters, accommodation_location → output: transport_matrix)
4. carbon_calculator: Calculates carbon emissions for transport routes using Climatiq API (input: transport_matrix → output: carbon_scored_transport_matrix)

TOOL WORKFLOW:
The expected flow is: places_api (with internal interest mapping and agent evaluation) → graph_generator → transport_analyzer → carbon_calculator
The agent directly maps interests to place types and evaluates places from places_api results based on user interests and preferences - no separate mapping or evaluation tools needed.

CONSTRAINTS & RULES:
- Maximum {current_state.get('max_iterations', 5)} iterations to avoid infinite loops
- Must achieve minimum {current_state.get('required_places', 15)} quality attractions (accommodation does NOT count toward this number)
- Focus on tourist attractions, avoid restaurants/food places
- Prioritize highly-rated places (4.0+ stars) with sufficient reviews
- Consider geographic distribution around accommodation
- Include accommodation in transport analysis and final output, but exclude from attraction count
- If insufficient results after tool usage, modify search parameters and retry
- Each tool usage should build upon previous results

DECISION FRAMEWORK:
Analyze the current situation and decide:
1. Do I have sufficient results to meet the goal? (Check: current count >= required count)
2. If insufficient, which tool should I use next and with what parameters?
3. Should I modify search strategy (expand radius, broaden interests, lower thresholds)?
4. Have I reached maximum iterations without success?

Based on the current context, what is your next action? Provide your reasoning and specific tool usage plan.

CURRENT SITUATION ANALYSIS NEEDED:
Current results: {len(current_state.get('current_results', []))} out of {current_state.get('required_places', 15)} required
"""

        return prompt
    
    def call_agent_llm(self, prompt: str) -> str:
        """Call LLM for agent reasoning and decision-making."""
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Fixed invalid model name
                    messages=[
                        {"role": "system", "content": "You are a Singapore Attraction Discovery Agent that makes strategic decisions about tool usage to achieve user goals. Respond with the exact format requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Less creative, more deterministic
                    max_completion_tokens=1500
                )
                return response.choices[0].message.content

            elif self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.1,  # Less creative, more deterministic - consistent with OpenAI
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            else:
                raise ValueError("No LLM client available for agent reasoning")
                
        except Exception as e:
            logger.error(f"Agent LLM reasoning failed: {e}")
            return "ERROR: Cannot reason about next actions"

    def parse_react_action(self, react_response: str) -> Dict[str, Any]:
        """
        Parse Action from LLM's ReAct response.

        Expected format:
        Action: {"tool": "tool_name", "parameters": {...}}
        """
        try:
            # Look for Action: line in the response
            lines = react_response.split('\n')
            action_line = None

            for line in lines:
                if line.strip().startswith('Action:'):
                    action_line = line.strip()
                    break

            if not action_line:
                logger.warning("No Action found in ReAct response")
                return {'error': 'No Action found in response'}

            # Extract JSON from Action: line
            json_start = action_line.find('{')
            if json_start == -1:
                logger.warning("No JSON found in Action line")
                return {'error': 'No JSON found in Action line'}

            json_str = action_line[json_start:]
            action_data = json.loads(json_str)

            logger.info(f"PARSED ACTION: {action_data}")
            return action_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Action JSON: {e}")
            return {'error': f'Invalid JSON in Action: {e}'}
        except Exception as e:
            logger.error(f"Error parsing ReAct action: {e}")
            return {'error': f'Failed to parse action: {e}'}

    def execute_tool_action(self, action_data: Dict[str, Any], user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specific tool action decided by the LLM.

        Args:
            action_data: Parsed action from ReAct response
            user_input: Original user input data

        Returns:
            Dictionary with execution result and observation
        """
        try:
            tool_name = action_data.get('tool')
            # Handle both 'parameters' and 'input' keys from LLM
            parameters = action_data.get('parameters', action_data.get('input', {}))

            if not tool_name or tool_name not in self.available_tools:
                return {
                    'error': f'Unknown tool: {tool_name}',
                    'observation': f'Error: Tool "{tool_name}" not available'
                }

            tool_instance = self.available_tools[tool_name]['instance']

            logger.info(f"EXECUTING TOOL: {tool_name} with parameters: {parameters}")

            # Execute specific tool based on name
# interest_mapper removed - mapping now handled internally in places_api execution
            if tool_name == 'places_api':
                place_types = parameters.get('place_types', [])
                accommodation_location = parameters.get('accommodation_location',
                                                       user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198}))
                max_distance_km = parameters.get('max_distance_km',
                                                self.agent_state.get('current_search_radius_km', 25))
                accessibility_needs = parameters.get('accessibility_needs',
                                                   user_input.get('accessibility_needs'))

                result = tool_instance._run(
                    place_types=place_types,
                    accommodation_location=accommodation_location,
                    accessibility_needs=accessibility_needs,
                    max_distance_km=max_distance_km
                )

# place_evaluator removed - evaluation now handled directly in places_api execution

            elif tool_name == 'graph_generator':
                evaluated_places = parameters.get('evaluated_places', [])
                result = tool_instance.generate_place_clusters_and_recommendations(
                    discovered_places=evaluated_places,
                    transport_connections=[],  # Will be filled after transport analysis
                    accommodation_location=user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})
                )

            elif tool_name == 'transport_analyzer':
                place_clusters = parameters.get('place_clusters', [])
                accommodation_location = parameters.get('accommodation_location',
                                                       user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198}))

                result = tool_instance._run(
                    recommended_places=place_clusters,
                    accommodation_location=accommodation_location
                )

            elif tool_name == 'carbon_calculator':
                transport_matrix = parameters.get('transport_matrix', [])
                result = tool_instance.calculate_multiple_routes(transport_matrix)

            else:
                return {
                    'error': f'Tool execution not implemented for: {tool_name}',
                    'observation': f'Error: Execution logic missing for tool "{tool_name}"'
                }

            # Format result as observation
            observation = f"Tool '{tool_name}' executed successfully. Result: {str(result)[:200]}..."
            if len(str(result)) > 200:
                observation += f" (truncated, full result available)"

            logger.info(f"TOOL RESULT: {tool_name} completed successfully")

            return {
                'result': result,
                'observation': observation,
                'tool_used': tool_name
            }

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"TOOL ERROR: {error_msg}")
            return {
                'error': error_msg,
                'observation': f"Error executing {tool_name}: {str(e)}"
            }
    
    def execute_agentic_workflow(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agentic workflow with iterative tool usage and refinement.
        
        The agent uses RGC prompting to decide which tools to use and when,
        iteratively refining results until the goal is achieved or max iterations reached.
        """
        
        # Initialize agent state
        self.agent_state['required_places'] = self.calculate_required_places(
            user_input['pace'], user_input['duration_days']
        )
        self.agent_state['current_results'] = []
        self.agent_state['iteration_count'] = 0
        
        # Set initial search radius from user input
        base_radius = user_input.get('optional', {}).get('max_distance_km', 25.0)
        self.agent_state['base_search_radius_km'] = base_radius
        self.agent_state['current_search_radius_km'] = base_radius
        
        logger.info(f"=== AGENT WORKFLOW STARTED ===")
        logger.info(f"Goal: Find {self.agent_state['required_places']} attractions for user interests: {user_input.get('interests', 'general')}")
        logger.info(f"Initial search radius: {self.agent_state['current_search_radius_km']}km")
        
        # Main agentic loop
        while self.agent_state['iteration_count'] < self.agent_state['max_iterations']:
            self.agent_state['iteration_count'] += 1
            
            logger.info(f"--- Iteration {self.agent_state['iteration_count']} ---")
            logger.info(f"Current search radius: {self.agent_state['current_search_radius_km']}km")
            
            # Check if we have sufficient results
            validation = self.validate_output_quantity(
                self.agent_state['current_results'], 
                self.agent_state['required_places']
            )
            
            if validation['sufficient']:
                logger.info(f"✅ Goal achieved! Found {validation['current_count']} places (required: {validation['required_count']})")
                break
            
            # Generate RGC prompt for agent reasoning
            rgc_prompt = self.create_rgc_prompt(user_input, self.agent_state)
            
            # Get agent's ReAct reasoning and decision
            agent_reasoning = self.call_agent_llm(rgc_prompt)
            
            # Display agent's ReAct response directly
            print(agent_reasoning)
            print("\n" + "-" * 40 + "\n")
            
            # Execute agent's decided actions based on LLM reasoning
            action_result = self._execute_llm_decided_actions(agent_reasoning, user_input)

            if action_result.get('error'):
                logger.error(f"Action execution failed: {action_result['error']}")
                self.agent_state['last_action'] = 'error'
                self.agent_state['last_result_summary'] = f"Tool execution failed: {action_result['error']}"
                continue

            # Save tool response to cache for LLM context (central location for all tools)
            if action_result.get('data') and action_result.get('action'):
                self.save_tool_response_to_cache(
                    action_result['action'],
                    self.agent_state['iteration_count'],
                    action_result['data']
                )

            # Update agent state with last action info
            tool_used = action_result.get('action', 'unknown_tool')
            self.agent_state['last_action'] = tool_used

            # Update results if new data was obtained
            new_count = 0
            if action_result.get('new_results'):
                initial_count = len(self.agent_state['current_results'])
                self.agent_state['current_results'].extend(action_result['new_results'])

                # Remove duplicates based on place_id
                seen_ids = set()
                unique_results = []
                for result in self.agent_state['current_results']:
                    place_id = result.get('place_id', '')
                    if place_id and place_id not in seen_ids:
                        seen_ids.add(place_id)
                        unique_results.append(result)
                self.agent_state['current_results'] = unique_results

                new_count = len(unique_results) - initial_count
                logger.info(f"Results updated: {len(self.agent_state['current_results'])} unique places (+{new_count} new)")

            # Store data for LLM context (especially for places_api string responses)
            if action_result.get('data'):
                self.agent_state['last_tool_data'] = action_result['data']
                logger.info(f"Stored tool data for LLM context: {len(str(action_result['data']))} characters")

            # Create simple result summary for LLM
            current_attraction_count = len([r for r in self.agent_state['current_results'] if r.get('place_id') != 'accommodation'])
            required_count = self.agent_state['required_places']

            if new_count > 0:
                self.agent_state['last_result_summary'] = f"Found {new_count} new attractions (total: {current_attraction_count}/{required_count})"
            elif action_result.get('data') and 'Found' in str(action_result['data']):
                # Handle places_api string data that contains places
                self.agent_state['last_result_summary'] = f"Tool returned places data - needs evaluation (current: {current_attraction_count}/{required_count})"
            else:
                self.agent_state['last_result_summary'] = f"Tool completed, no new attractions added (current: {current_attraction_count}/{required_count})"
        
        # Final validation
        final_validation = self.validate_output_quantity(
            self.agent_state['current_results'],
            self.agent_state['required_places']
        )
        
        logger.info(f"=== AGENT WORKFLOW COMPLETED ===")
        logger.info(f"Final results: {final_validation['current_count']}/{final_validation['required_count']} places")
        logger.info(f"Success rate: {final_validation['success_rate']:.1%}")
        
        return {
            'results': self.agent_state['current_results'],
            'validation': final_validation,
            'iterations_used': self.agent_state['iteration_count'],
            'success': final_validation['sufficient']
        }
    
    def _execute_llm_decided_actions(self, agent_reasoning: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM reasoning and execute the tool actions the agent decided to use.

        Extracts tool decisions from agent's ReAct reasoning and executes them.
        """

        try:
            # Parse which tool the LLM decided to use from the Action line
            tool_decision = self._parse_tool_decision(agent_reasoning)

            if tool_decision.get('error'):
                logger.error(f"Failed to parse tool decision: {tool_decision['error']}")
                return tool_decision

            tool_name = tool_decision['tool']
            logger.info(f"LLM DECIDED TO USE TOOL: {tool_name}")

            # Execute the specific tool based on LLM's decision
# interest_mapper removed - mapping now handled internally in places_api execution
            if tool_name == 'places_api':
                return self._execute_places_search(user_input, tool_decision.get('parameters', {}))

# place_evaluator removed - evaluation now handled directly in places_api execution

            elif tool_name == 'graph_generator':
                return self._execute_clustering(user_input, tool_decision.get('parameters', {}))

            elif tool_name == 'transport_analyzer':
                return self._execute_transport_analysis(user_input, tool_decision.get('parameters', {}))

            elif tool_name == 'carbon_calculator':
                return self._execute_carbon_scoring(user_input, tool_decision.get('parameters', {}))

            else:
                error_msg = f"Unknown tool decision: {tool_name}"
                logger.error(error_msg)
                return {'error': error_msg}

        except Exception as e:
            logger.error(f"Error executing LLM decided actions: {e}")
            return {'error': str(e)}

    def _parse_tool_decision(self, agent_reasoning: str) -> Dict[str, Any]:
        """
        Parse tool decision from LLM's ReAct reasoning.

        Looks for Action: lines and extracts tool name and parameters.
        """
        try:
            lines = agent_reasoning.split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith('Action:'):
                    # Extract tool name from Action line
                    # Support formats like:
                    # Action: interest_mapper
                    # Action: places_api with radius 30km
                    # Action: {"tool": "places_api", "parameters": {...}}

                    action_content = line[7:].strip()  # Remove "Action:" prefix

                    # Try JSON format first
                    if action_content.startswith('{'):
                        try:
                            return json.loads(action_content)
                        except json.JSONDecodeError:
                            pass

                    # Parse simple tool name format
                    tool_name = action_content.split()[0]  # First word is tool name

                    # Check if it's a valid tool
                    if tool_name in self.available_tools:
                        logger.info(f"PARSED TOOL DECISION: {tool_name}")
                        return {'tool': tool_name, 'parameters': {}}

                    # Try matching partial tool names
                    for available_tool in self.available_tools.keys():
                        if tool_name in available_tool or available_tool in tool_name:
                            logger.info(f"MATCHED TOOL DECISION: {available_tool} (from '{tool_name}')")
                            return {'tool': available_tool, 'parameters': {}}

            # No clear action found - log the reasoning for debugging
            print(f"DEBUG - Full LLM response:\n{agent_reasoning}\n" + "="*50)
            logger.warning(f"No Action: line found in LLM response. Full response: {agent_reasoning[:500]}...")
            return {'error': 'No valid tool decision found in agent reasoning'}

        except Exception as e:
            logger.error(f"Error parsing tool decision: {e}")
            return {'error': f'Failed to parse tool decision: {e}'}

    def _execute_interest_mapping(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute interest mapping tool."""
        try:
            interests = user_input.get('interests', [])
            if isinstance(interests, str):
                interests = [interests]

            logger.info(f"TOOL CALL: interest_mapper with interests={interests}")
            result = self.available_tools['interest_mapper']['instance'].map_interests_to_place_types(interests)
            logger.info(f"TOOL RESULT: interest_mapper returned place_types={result}")

            return {'new_results': [], 'action': 'interest_mapping', 'data': result}

        except Exception as e:
            logger.error(f"Interest mapping failed: {e}")
            return {'error': f"Interest mapping failed: {e}"}

    def _execute_places_search(self, user_input: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute places API search tool with internal interest mapping."""
        try:
            # Map user interests to place types internally (no separate tool needed)
            user_interests = user_input.get('interests', [])
            if isinstance(user_interests, str):
                user_interests = [user_interests]

            place_types = self.map_interests_to_place_types(user_interests)
            accommodation_location = user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})
            search_radius = parameters.get('max_distance_km', self.agent_state['current_search_radius_km'])
            accessibility_needs = user_input.get('accessibility_needs')

            logger.info(f"AGENT MAPPING: {user_interests} → {place_types}")
            logger.info(f"TOOL CALL: places_api with place_types={place_types}, radius={search_radius}km")

            result = self.available_tools['places_api']['instance']._run(
                place_types=place_types,
                accommodation_location=accommodation_location,
                accessibility_needs=accessibility_needs,
                max_distance_km=search_radius
            )
            logger.info(f"TOOL RESULT: places_api returned data")

            # For now, return raw result - parsing can be added later
            # TODO: Parse the string result to extract place data
            new_results = []  # Temporary - will fix parsing
            logger.info(f"PARSED: {len(new_results)} places from API response")

            return {'new_results': new_results, 'action': 'places_search_with_mapping', 'data': result}

        except Exception as e:
            logger.error(f"Places search failed: {e}")
            return {'error': f"Places search failed: {e}"}

    def _execute_place_evaluation(self, user_input: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute place evaluation tool."""
        try:
            # Get places data from last tool result or parameters
            raw_places_data = parameters.get('raw_places_data', '') or self.agent_state.get('last_tool_data', '')
            user_interests = user_input.get('interests', [])
            user_preferences = user_input.get('user_preferences', {})

            logger.info(f"TOOL CALL: place_evaluator with interests={user_interests}")
            result = self.available_tools['place_evaluator']['instance']._run(
                places_data=raw_places_data,
                user_interests=user_interests,
                user_preferences=user_preferences
            )
            logger.info(f"TOOL RESULT: place_evaluator completed evaluation")

            # Parse results (simplified)
            new_results = []  # Would contain evaluated place dictionaries
            return {'new_results': new_results, 'action': 'place_evaluation', 'data': result}

        except Exception as e:
            logger.error(f"Place evaluation failed: {e}")
            return {'error': f"Place evaluation failed: {e}"}

    def _execute_clustering(self, user_input: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute clustering tool."""
        try:
            evaluated_places = parameters.get('evaluated_places', self.agent_state.get('current_results', []))
            accommodation_location = user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})

            logger.info(f"TOOL CALL: graph_generator with {len(evaluated_places)} places")
            result = self.available_tools['graph_generator']['instance'].generate_place_clusters_and_recommendations(
                discovered_places=evaluated_places,
                transport_connections=[],
                accommodation_location=accommodation_location
            )
            logger.info(f"TOOL RESULT: graph_generator created clusters")

            return {'new_results': [], 'action': 'clustering', 'data': result}

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {'error': f"Clustering failed: {e}"}

    def _execute_transport_analysis(self, user_input: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transport analysis tool."""
        try:
            place_clusters = parameters.get('place_clusters', [])
            accommodation_location = user_input.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})

            logger.info(f"TOOL CALL: transport_analyzer with clusters")
            result = self.available_tools['transport_analyzer']['instance']._run(
                recommended_places=place_clusters,
                accommodation_location=accommodation_location
            )
            logger.info(f"TOOL RESULT: transport_analyzer completed analysis")

            return {'new_results': [], 'action': 'transport_analysis', 'data': result}

        except Exception as e:
            logger.error(f"Transport analysis failed: {e}")
            return {'error': f"Transport analysis failed: {e}"}

    def _execute_carbon_scoring(self, user_input: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute carbon scoring tool."""
        try:
            transport_matrix = parameters.get('transport_matrix', [])

            logger.info(f"TOOL CALL: carbon_calculator with transport data")
            result = self.available_tools['carbon_calculator']['instance'].calculate_multiple_routes(transport_matrix)
            logger.info(f"TOOL RESULT: carbon_calculator completed scoring")

            return {'new_results': [], 'action': 'carbon_scoring', 'data': result}

        except Exception as e:
            logger.error(f"Carbon scoring failed: {e}")
            return {'error': f"Carbon scoring failed: {e}"}

    def _execute_initial_search(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute initial search: interest mapping + places API."""
        try:
            # Step 1: Map interests
            interests = user_input.get('interests', 'general sightseeing')
            if isinstance(interests, str):
                interests = [interests]
            
            logger.info(f"TOOL CALL: interest_mapper.map_interests_to_place_types(interests={interests})")
            place_types = self.available_tools['interest_mapper']['instance'].map_interests_to_place_types(interests)
            logger.info(f"TOOL RESULT: interest_mapper returned place_types={place_types}")
            
            # Step 2: Search places
            optional_data = user_input.get('optional', {})
            accommodation_location = optional_data.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})
            accessibility_needs = optional_data.get('accessibility_needs')
            
            # Use current search radius from agent state
            search_radius = self.agent_state['current_search_radius_km']
            
            logger.info(f"TOOL CALL: places_api._run(place_types={place_types}, accommodation_location={accommodation_location}, max_distance_km={search_radius})")
            places_data = self.available_tools['places_api']['instance']._run(
                place_types=place_types,
                accommodation_location=accommodation_location,
                accessibility_needs=accessibility_needs,
                max_distance_km=search_radius
            )
            logger.info(f"TOOL RESULT: places_api returned {len(places_data)} characters of place data")
            
            # Parse places data (simplified - in real implementation, you'd parse the text response)
            new_results = []  # This would contain parsed place dictionaries
            logger.info(f"AGENT PROCESSING: Parsed {len(new_results)} places from places_api response")
            
            return {'new_results': new_results, 'action': 'initial_search'}
            
        except Exception as e:
            logger.error(f"TOOL ERROR: Initial search failed: {e}")
            return {'error': f"Initial search failed: {e}"}
    
    def _execute_expanded_search(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute expanded search with broader parameters."""
        try:
            # Expand search radius progressively based on iteration
            optional_data = user_input.get('optional', {})
            accommodation_location = optional_data.get('accommodation_location', {'lat': 1.3521, 'lon': 103.8198})
            
            # Progressive expansion: +50% per iteration beyond first
            current_radius = self.agent_state['current_search_radius_km']
            base_radius = self.agent_state['base_search_radius_km']
            iteration = self.agent_state['iteration_count']
            
            # Expand radius: iteration 2 = +50%, iteration 3 = +100%, etc.
            expansion_factor = 1.0 + (0.5 * (iteration - 1))
            expanded_distance = base_radius * expansion_factor
            
            # Update agent state with new radius
            self.agent_state['current_search_radius_km'] = expanded_distance
            
            logger.info(f"AGENT DECISION: Expanding search radius from {current_radius}km to {expanded_distance}km (iteration {iteration}, factor: {expansion_factor:.1f}x)")
            
            # Use broader place types
            broad_place_types = ['tourist_attraction', 'park', 'museum', 'art_gallery', 'zoo', 'amusement_park']
            logger.info(f"AGENT DECISION: Using broader place types: {broad_place_types}")
            
            logger.info(f"TOOL CALL: places_api._run(place_types={broad_place_types}, max_distance_km={expanded_distance})")
            places_data = self.available_tools['places_api']['instance']._run(
                place_types=broad_place_types,
                accommodation_location=accommodation_location,
                accessibility_needs=optional_data.get('accessibility_needs'),
                max_distance_km=expanded_distance
            )
            logger.info(f"TOOL RESULT: places_api returned {len(places_data)} characters of expanded place data")
            
            new_results = []  # Parse results
            logger.info(f"AGENT PROCESSING: Parsed {len(new_results)} additional places from expanded search")
            
            return {'new_results': new_results, 'action': 'expanded_search'}
            
        except Exception as e:
            logger.error(f"TOOL ERROR: Expanded search failed: {e}")
            return {'error': f"Expanded search failed: {e}"}
    
    def _execute_llm_evaluation(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM evaluation of current results."""
        try:
            if not self.agent_state['current_results']:
                logger.warning("AGENT ERROR: No results to evaluate")
                return {'error': 'No results to evaluate'}
            
            # Create places data string from current results
            places_data_str = self._format_places_for_llm(self.agent_state['current_results'])
            logger.info(f"AGENT PROCESSING: Formatted {len(self.agent_state['current_results'])} places for LLM evaluation")
            
            # Build user preferences
            user_preferences = {
                'budget': user_input.get('budget'),
                'pace': user_input.get('pace')
            }
            
            interests = user_input.get('interests', 'general sightseeing')
            if isinstance(interests, str):
                interests = [interests]
            
            logger.info(f"TOOL CALL: place_evaluator._run(user_interests={interests}, max_places={self.agent_state['required_places']})")
            
            # Use LLM evaluator
            evaluation_result = self.available_tools['place_evaluator']['instance']._run(
                places_data=places_data_str,
                user_interests=interests,
                user_preferences=user_preferences,
                max_places=self.agent_state['required_places']
            )
            logger.info(f"TOOL RESULT: place_evaluator returned {len(evaluation_result)} characters of evaluation data")
            
            # Parse evaluation result
            new_results = []  # Parse LLM evaluation results
            logger.info(f"AGENT PROCESSING: Extracted {len(new_results)} refined recommendations from LLM evaluation")
            
            return {'new_results': new_results, 'action': 'llm_evaluation'}
            
        except Exception as e:
            logger.error(f"TOOL ERROR: LLM evaluation failed: {e}")
            return {'error': f"LLM evaluation failed: {e}"}
    
    def _format_places_for_llm(self, places: List[Dict]) -> str:
        """Format current places data for LLM evaluation."""
        if not places:
            return "No places data available."
        
        formatted = "Current Places Data:\n\n"
        for i, place in enumerate(places, 1):
            formatted += f"{i}. {place.get('name', 'Unknown')}\n"
            formatted += f"   Type: {place.get('type', 'Unknown')}\n"
            formatted += f"   Rating: {place.get('rating', 0)}/5.0\n"
            formatted += f"   Location: {place.get('vicinity', 'Unknown')}\n\n"
        
        return formatted

def main(input_file: str):
    """Main agentic workflow for Singapore attraction discovery."""
    # Load environment variables
    load_dotenv()
    
    # Load input data
    input_data = load_input_file(input_file)
    
    print("Singapore Attraction Discovery Agent")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Trip duration: {input_data['duration_days']} days")
    print(f"Budget: ${input_data['budget']}")
    print(f"Pace: {input_data['pace']}")
    # interests = input_data.get("optional", {}).get("interests", "undefined")
    optional_data = input_data.get("optional", {})
    interests = optional_data.get("interests", "general sightseeing")
    print(f"Interests: {interests}")
    print("\n" + "=" * 50 + "\n")
    
    try:
        # Initialize the agentic system
        agent = SingaporeAttractionAgent()
        
        # Apply existing defaults for undefined data
        # Default accommodation location to Singapore's center if not provided
        default_location = {"lat": 1.3521, "lon": 103.8198}  # Singapore center
        accommodation_location = optional_data.get("accommodation_location", default_location)
        
        # Default interests to general sightseeing (maps to tourist_attraction)
        if not interests or interests == "undefined":
            interests = "general sightseeing"  # This will map to tourist_attraction
        
        # Default search radius to 25km (from SEARCH_RADIUS_METERS=25000 in environment)
        default_radius_km = float(os.getenv("SEARCH_RADIUS_METERS")) / 1000.0  # Convert meters to km
        if "max_distance_km" not in optional_data:
            optional_data["max_distance_km"] = default_radius_km
            
        # Ensure optional data has accommodation location
        if "accommodation_location" not in optional_data:
            optional_data["accommodation_location"] = default_location
        
        # Prepare user input for agent with defaults applied
        user_input = {
            'pace': input_data['pace'],
            'duration_days': input_data['duration_days'],
            'budget': input_data['budget'],
            'interests': interests,
            'optional': optional_data,
            'accommodation_location': accommodation_location,
            'accessibility_needs': optional_data.get('accessibility_needs')
        }
        
        # Build user preferences from available data (keep existing pattern)
        user_preferences = {
            "budget": input_data["budget"],
            "pace": input_data["pace"]
        }
        
        # Add optional preferences if they exist (keep existing logic)
        optional_prefs = ["group_type", "eco_preferences", "dietary_preferences"]
        for pref in optional_prefs:
            if pref in optional_data:
                user_preferences[pref] = optional_data[pref]
        
        # Add user_preferences to user_input for agent
        user_input['user_preferences'] = user_preferences
        
        # Execute agentic workflow
        print("Initializing Singapore Attraction Discovery Agent...")
        print("Agent will use RGC prompting and iterative tool usage to find optimal attractions")
        print(f"Target: Find attractions for '{interests}' with {input_data['pace']} pace")
        print(f"Base location: {accommodation_location['lat']:.4f}, {accommodation_location['lon']:.4f}")
        print("\n" + "-" * 60 + "\n")
        
        workflow_result = agent.execute_agentic_workflow(user_input)
        
        # Display final results
        print("\n" + "=" * 60)
        print("AGENT WORKFLOW RESULTS")
        print("=" * 60)
        
        if workflow_result['success']:
            print("SUCCESS: Agent achieved goal!")
            print(f"   Found: {len(workflow_result['results'])} attractions")
            print(f"   Required: {workflow_result['validation']['required_count']}")
            print(f"   Success Rate: {workflow_result['validation']['success_rate']:.1%}")
        else:
            print("PARTIAL SUCCESS: Agent made progress but didn't fully achieve goal")
            print(f"   Found: {len(workflow_result['results'])} attractions")
            print(f"   Required: {workflow_result['validation']['required_count']}")
            print(f"   Deficit: {workflow_result['validation']['deficit']} more needed")
        
        print(f"   Iterations Used: {workflow_result['iterations_used']}/5")
        print(f"   Agent Reasoning Log: agent_reasoning.log")
        
        # Display attraction results (simplified)
        if workflow_result['results']:
            print(f"\nDISCOVERED ATTRACTIONS:")
            print("-" * 40)
            for i, place in enumerate(workflow_result['results'][:10], 1):  # Show first 10
                print(f"{i}. {place.get('name', 'Unknown')}")
                print(f"   Type: {place.get('type', 'N/A')}")
                print(f"   Rating: {place.get('rating', 'N/A')}/5.0")
                print()
            
            if len(workflow_result['results']) > 10:
                print(f"   ... and {len(workflow_result['results']) - 10} more attractions")
        
        print(f"\nAgent Performance Summary:")
        print(f"   - Goal Achievement: {'SUCCESS' if workflow_result['success'] else 'PARTIAL'}")
        print(f"   - Tool Usage Efficiency: {workflow_result['iterations_used']}/5 iterations")
        print(f"   - Result Quality: {workflow_result['validation']['success_rate']:.0%} of target")

        # Create output.py file with the results
        output_data = {
            'workflow_result': workflow_result,
            'user_input': user_input,
            'agent_performance': {
                'goal_achievement': 'SUCCESS' if workflow_result['success'] else 'PARTIAL',
                'tool_usage_efficiency': f"{workflow_result['iterations_used']}/5 iterations",
                'result_quality': f"{workflow_result['validation']['success_rate']:.0%} of target"
            }
        }

        # Write to output.py
        with open('output.py', 'w') as f:
            f.write("# Singapore Attraction Discovery Agent Output\n")
            f.write("# Generated automatically by main.py\n\n")
            f.write(f"output_data = {json.dumps(output_data, indent=4, default=str)}\n")

        print(f"\nOutput written to: output.py")

    except Exception as e:
        logger.error(f"Agent workflow failed: {e}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def show_usage():
    """Display usage information."""
    print("Usage: python main.py <input_file>")
    print("\nExample:")
    print("  python main.py inputs/simple_input.json")
    print("  python main.py inputs/complex_input.json")
    print("\nAvailable input files:")
    
    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    if os.path.exists(inputs_dir):
        for file in sorted(os.listdir(inputs_dir)):
            if file.endswith('.json'):
                print(f"  inputs/{file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        show_usage()
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Handle relative paths from inputs directory
    if not input_file.startswith('inputs/') and not os.path.exists(input_file):
        test_path = f"inputs/{input_file}"
        if os.path.exists(test_path):
            input_file = test_path
    
    main(input_file)