"""
Comprehensive Interest and Uninterest Mapping Configuration
Maps user preferences (both positive and negative) to Google Places API v1 types
"""

# ============================================================================
# UNINTEREST MAPPINGS - What users want to avoid
# ============================================================================

UNINTEREST_MAPPINGS = {
    # ========== DIETARY RESTRICTIONS ==========
    "no meat": ["steak_house", "barbecue_restaurant", "hamburger_restaurant"],
    "no beef": ["steak_house", "hamburger_restaurant"],
    "no pork": ["barbecue_restaurant"],
    "no seafood": ["seafood_restaurant", "sushi_restaurant"],
    "no shellfish": ["seafood_restaurant", "sushi_restaurant"],
    "no alcohol": ["bar", "pub", "wine_bar", "night_club", "liquor_store"],
    "no gluten": ["bakery", "pizza_restaurant", "sandwich_shop"],
    "no dairy": ["ice_cream_shop", "dessert_shop"],
    "no nuts": ["bakery", "dessert_shop"],
    "no spicy": ["mexican_restaurant", "indian_restaurant", "thai_restaurant", "korean_restaurant"],
    "halal only": ["bar", "pub", "wine_bar", "night_club"],
    "kosher only": ["seafood_restaurant"],
    
    # ========== LIFESTYLE PREFERENCES ==========
    # Avoiding party/loud environments
    "no nightlife": ["night_club", "bar", "pub", "casino"],
    "no clubs": ["night_club"],
    "no bars": ["bar", "pub", "wine_bar"],
    "no loud": ["night_club", "bar", "concert_hall", "amphitheatre"],
    "no party": ["night_club", "bar", "casino"],
    "no gambling": ["casino"],
    "no smoking": ["bar", "pub", "night_club", "casino"],
    "quiet": ["night_club", "bar", "amusement_park", "concert_hall"],
    
    # Avoiding crowds
    "no crowds": ["amusement_park", "shopping_mall", "tourist_attraction", "night_club"],
    "no touristy": ["tourist_attraction", "gift_shop"],
    "no mainstream": ["shopping_mall", "fast_food_restaurant"],
    "no chains": ["fast_food_restaurant"],
    
    # ========== ACTIVITY PREFERENCES ==========
    "no outdoor": ["park", "hiking_area", "beach", "campground", "zoo"],
    "no indoor": ["museum", "shopping_mall", "movie_theater"],
    "no walking": ["hiking_area", "park", "zoo"],
    "no hiking": ["hiking_area", "national_park"],
    "no sports": ["sports_complex", "stadium", "gym", "fitness_center"],
    "no swimming": ["swimming_pool", "beach", "water_park"],
    "no adventure": ["adventure_sports_center", "amusement_park", "ski_resort"],
    "no extreme": ["adventure_sports_center", "roller_coaster"],
    
    # ========== FAMILY/KIDS PREFERENCES ==========
    "no kids": ["playground", "childrens_camp", "amusement_park"],
    "adults only": ["playground", "childrens_camp", "zoo", "aquarium"],
    "no family": ["amusement_park", "zoo", "aquarium", "playground"],
    "kid free": ["playground", "childrens_camp", "amusement_park"],
    
    # ========== BUDGET PREFERENCES ==========
    "no expensive": ["fine_dining_restaurant", "resort_hotel", "spa", "casino"],
    "no luxury": ["fine_dining_restaurant", "resort_hotel", "spa"],
    "budget": ["fine_dining_restaurant", "resort_hotel"],
    "no shopping": ["shopping_mall", "department_store", "clothing_store", "jewelry_store"],
    
    # ========== HEALTH & WELLNESS ==========
    "no spa": ["spa", "wellness_center", "massage"],
    "no wellness": ["wellness_center", "yoga_studio", "spa"],
    "no gym": ["gym", "fitness_center", "sports_complex"],
    
    # ========== CULTURAL/RELIGIOUS ==========
    "no religious": ["church", "hindu_temple", "mosque", "synagogue"],
    "no temples": ["hindu_temple", "buddhist_temple"],
    "no churches": ["church"],
    "no mosques": ["mosque"],
    
    # ========== SPECIFIC CUISINES TO AVOID ==========
    "no asian": ["asian_restaurant", "chinese_restaurant", "japanese_restaurant", "korean_restaurant", "thai_restaurant"],
    "no chinese": ["chinese_restaurant"],
    "no japanese": ["japanese_restaurant", "sushi_restaurant", "ramen_restaurant"],
    "no indian": ["indian_restaurant"],
    "no italian": ["italian_restaurant", "pizza_restaurant"],
    "no mexican": ["mexican_restaurant"],
    "no american": ["american_restaurant", "hamburger_restaurant", "steak_house"],
    "no fast food": ["fast_food_restaurant", "hamburger_restaurant"],
    
    # ========== ANIMALS/NATURE ==========
    "no animals": ["zoo", "aquarium", "wildlife_park", "pet_store"],
    "no zoo": ["zoo", "wildlife_park"],
    "no nature": ["park", "national_park", "botanical_garden", "hiking_area"],
    "no parks": ["park", "national_park", "state_park"],
    
    # ========== ENTERTAINMENT ==========
    "no movies": ["movie_theater"],
    "no theater": ["movie_theater", "performing_arts_theater"],
    "no museums": ["museum", "art_gallery"],
    "no art": ["art_gallery", "art_studio", "museum"],
    "no games": ["video_arcade", "amusement_center", "casino"],
    "no arcade": ["video_arcade", "amusement_center"],
}

# ============================================================================
# SPECIAL INTEREST CATEGORIES - Complex interest combinations
# ============================================================================

SPECIAL_INTEREST_CATEGORIES = {
    "romantic": {
        "include": ["fine_dining_restaurant", "wine_bar", "observation_deck", "park", "beach", "spa"],
        "exclude": ["fast_food_restaurant", "playground", "childrens_camp"]
    },
    "business": {
        "include": ["restaurant", "cafe", "bar", "hotel"],
        "exclude": ["night_club", "amusement_park", "playground"]
    },
    "adventure": {
        "include": ["adventure_sports_center", "hiking_area", "ski_resort", "campground", "national_park"],
        "exclude": ["shopping_mall", "museum"]
    },
    "cultural": {
        "include": ["museum", "art_gallery", "historical_landmark", "cultural_center", "performing_arts_theater"],
        "exclude": ["night_club", "casino", "fast_food_restaurant"]
    },
    "luxury": {
        "include": ["fine_dining_restaurant", "resort_hotel", "spa", "wellness_center", "wine_bar"],
        "exclude": ["fast_food_restaurant", "hostel", "convenience_store"]
    },
    "budget": {
        "include": ["fast_food_restaurant", "hostel", "park", "beach", "hiking_area"],
        "exclude": ["fine_dining_restaurant", "resort_hotel", "spa"]
    },
    "local": {
        "include": ["market", "food_court", "park"],
        "exclude": ["tourist_attraction", "gift_shop"]
    },
    "photography": {
        "include": ["observation_deck", "park", "beach", "historical_landmark", "botanical_garden"],
        "exclude": ["night_club", "casino"]
    }
}

# ============================================================================
# DIETARY EXCLUSIONS - Map dietary restrictions to excluded place types
# ============================================================================

DIETARY_EXCLUSIONS = {
    # Vegetarian - exclude meat-focused restaurants
    "vegetarian": [
        "steakhouse", "steak_house", "american_restaurant",
        "hamburger_restaurant", "barbecue_restaurant"
    ],

    # Vegan - exclude all animal products
    "vegan": [
        "steakhouse", "steak_house", "american_restaurant",
        "hamburger_restaurant", "barbecue_restaurant",
        "seafood_restaurant", "sushi_restaurant"
    ],

    # Halal - exclude pork and alcohol-focused places
    "halal": [
        "bar", "wine_bar", "cocktail_bar", "pub",
        "barbecue_restaurant"  # Often serves pork
    ],

    # Kosher - exclude non-kosher places
    "kosher": [
        "seafood_restaurant",  # Shellfish not kosher
        "barbecue_restaurant"  # Mixed meat types
    ],

    # Gluten-free - exclude bakery-heavy places
    "gluten-free": ["bakery", "pizza_restaurant"],
    "gluten free": ["bakery", "pizza_restaurant"],

    # Pescatarian - exclude meat but allow fish
    "pescatarian": [
        "steakhouse", "steak_house", "american_restaurant",
        "hamburger_restaurant", "barbecue_restaurant"
    ],

    # Dairy-free/Lactose intolerant
    "dairy-free": ["ice_cream_shop", "dessert_shop"],
    "dairy free": ["ice_cream_shop", "dessert_shop"],
    "lactose intolerant": ["ice_cream_shop", "dessert_shop"],

    # Nut allergy - exclude nut-heavy cuisines
    "nut allergy": ["thai_restaurant", "vietnamese_restaurant"],
    "nut-free": ["thai_restaurant", "vietnamese_restaurant"]
}

# Tag extraction prompts for LLM
TAG_EXTRACTION_PROMPT = """Extract 3 highly relevant descriptive tags for this Singapore location.

Place: {name}
Type: {place_type}
Description: {description}
Location: {location}
Price Level: {price_level}
Rating: {rating}
Existing tags: {existing_tags}

Requirements:
- New tags only (not in existing tags)
- Lowercase with hyphens (e.g., "family-friendly")
- Focus on: accessibility, timing, atmosphere, budget or unique features
- Consider local context when relevant

Return JSON array: ["tag1", "tag2", "tag3"]
"""

TAG_EXTRACTION_SYSTEM_PROMPT = """You are a travel data specialist who extracts actionable tags from place descriptions. 
You analyze places to identify key characteristics that help travelers make quick decisions. 
Always output valid JSON arrays containing 2-3 lowercase hyphenated tags."""

PLACE_DESCRIPTION_PROMPT = """Generate a 3-4 sentence description for this Singapore location.

Place: {name}
Address: {address}
Type: {place_type}

Include:
1. Location & accessibility (nearest MRT/landmarks)
2. Character & typical visitors (what makes it special, best times)
3. Practical tip (payment, booking, local advice)

Context hints:
- CBD: office crowds, lunch rush 12-2pm
- Heartlands: local prices, family-friendly
- Tourist areas: higher prices, crowded weekends

Example tone: "Located 5 minutes from Chinatown MRT, this heritage hawker..."

Write naturally and informatively."""

DESCRIPTION_SYSTEM_PROMPT = """You are a local travel guide who creates informative place descriptions for visitors. 
You write 3-4 sentences covering location context, key characteristics, and practical tips. 
Your tone is friendly and conversational, as if advising a friend."""

# LLM Configuration
LLM_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens_tags": 50
}

# Tag generation limits
TAG_LIMITS = {
    "max_total_tags": 8,
    "max_llm_tags": 8
}

# Rating-based tag thresholds
RATING_TAG_THRESHOLDS = {
    "highly_rated": {
        "min_rating": 4.4,
        "min_reviews": 100
    },
    "top_rated": {
        "min_rating": 4.7,
        "min_reviews": 500
    }
}

# ============================================================================
# AUTO-EXCLUSIONS - Types to automatically exclude for specific interests
# ============================================================================

# Map interests to types that should be EXCLUDED from searches
# Prevents irrelevant results (e.g., pet stores when searching for aquariums)
INTEREST_AUTO_EXCLUSIONS = {
    "aquarium": ["pet_store"],  # Aquarium attractions, not pet shops
    "zoo": ["pet_store"],        # Zoo attractions, not pet shops
    "wildlife": ["pet_store"],   # Wildlife viewing, not pet shops
    "animals": ["pet_store"],    # Animal attractions, not pet shops (unless they want pet shops)
}

# ============================================================================
# ATTRACTION RECOGNITION - Set of all Google Place types that are attraction-related
# ============================================================================
ATTRACTION_PLACE_TYPES = {
    # Business
    "farm", "ranch",
    
    # Culture
    "art_gallery", "art_studio", "auditorium", "cultural_landmark", "historical_place",
    "monument", "museum", "performing_arts_theater", "sculpture", "landmark", "point_of_interest",

    # Education
    "university",
    
    # Entertainment and Recreation
    "adventure_sports_center", "amphitheatre", "amusement_center", "amusement_park", "aquarium",
    "barbecue_area", "botanical_garden", "bowling_alley", "casino", "childrens_camp",
    "comedy_club", "community_center", "concert_hall", "convention_center", "cultural_center",
    "cycling_park", "dog_park", "ferris_wheel", "garden", "hiking_area", "historical_landmark",
    "internet_cafe", "karaoke", "marina", "movie_theater", "national_park", "night_club",
    "observation_deck", "opera_house", "park", "philharmonic_hall", "picnic_ground", "planetarium", "plaza",
    "roller_coaster", "skateboard_park", "state_park", "tourist_attraction", "video_arcade", "visitor_center", "water_park",
    "wildlife_park", "wildlife_refuge", "zoo",
    
    # Health and Wellness
    "sauna", "spa",

    # Natural Features
    "beach", "natural_feature",
    
    # Places of Worship
    "church", "hindu_temple", "mosque", "synagogue","place_of_worship",
    
    # Shopping
    "asian_grocery_store", "auto_parts_store", "bicycle_store", "book_store", "butcher_shop",
    "cell_phone_store", "clothing_store", "convenience_store", "department_store", "discount_store",
    "electronics_store", "food_store", "furniture_store", "gift_shop", "grocery_store",
    "hardware_store", "home_goods_store", "home_improvement_store", "jewelry_store",
    "liquor_store", "market", "pet_store", "shoe_store", "shopping_mall",
    "sporting_goods_store", "store", "supermarket", "warehouse_store", "wholesaler"
    
    # Sports
    "arena", "athletic_field", "fishing_charter", "fishing_pond", "fitness_center",
    "golf_course", "gym", "ice_skating_rink", "playground", "ski_resort",
    "sports_activity_location", "sports_club", "sports_coaching",
    "sports_complex", "stadium", "swimming_pool",
    
    # Transportation
    "international_airport"
}

# ============================================================================
# FOOD PLACE RECOGNITION - Set of all Google Place types that are food-related
# ============================================================================

# Complete set of food-related place types for classification
FOOD_PLACE_TYPES = {
    # Restaurants by cuisine
    "restaurant", "chinese_restaurant", "japanese_restaurant", "korean_restaurant",
    "thai_restaurant", "vietnamese_restaurant", "indian_restaurant", "indonesian_restaurant",
    "italian_restaurant", "french_restaurant", "mexican_restaurant", "spanish_restaurant",
    "greek_restaurant", "mediterranean_restaurant", "middle_eastern_restaurant",
    "american_restaurant", "seafood_restaurant", "asian_restaurant",

    # Restaurants by meal type
    "breakfast_restaurant", "brunch_restaurant", "fine_dining_restaurant",
    "fast_food_restaurant", "buffet_restaurant",

    # Restaurants by dietary
    "vegetarian_restaurant", "vegan_restaurant",

    # Specific food types
    "sushi_restaurant", "ramen_restaurant", "pizza_restaurant",
    "hamburger_restaurant", "steak_house", "barbecue_restaurant",
    "sandwich_shop", "deli",

    # Cafes & Coffee
    "cafe", "coffee_shop", "bistro",

    # Bakeries & Desserts
    "bakery", "dessert_shop", "ice_cream_shop", "candy_store",

    # Food courts & Markets
    "food_court", "hawker_centre", "night_market",

    # Bars & Drinks
    "bar", "wine_bar", "cocktail_bar", "pub", "bar_and_grill",

    # Food services
    "meal_takeaway", "meal_delivery", "food",

    # Other dining
    "diner", "tea_house"
}

# Food type mappings - Extract food-specific tags from Google Place types
FOOD_TYPE_MAPPINGS = {
    # Cuisines
    "chinese_restaurant": "chinese",
    "japanese_restaurant": "japanese",
    "korean_restaurant": "korean",
    "thai_restaurant": "thai",
    "vietnamese_restaurant": "vietnamese",
    "indian_restaurant": "indian",
    "italian_restaurant": "italian",
    "french_restaurant": "french",
    "mexican_restaurant": "mexican",
    "spanish_restaurant": "spanish",
    "greek_restaurant": "greek",
    "mediterranean_restaurant": "mediterranean",
    "middle_eastern_restaurant": "middle-eastern",
    "american_restaurant": "american",
    "seafood_restaurant": "seafood",

    # Meal types
    "breakfast_restaurant": "breakfast",
    "brunch_restaurant": "brunch",
    "fine_dining_restaurant": "fine-dining",
    "fast_food_restaurant": "fast-food",

    # Dietary
    "vegetarian_restaurant": "vegetarian",
    "vegan_restaurant": "vegan",

    # Specific foods
    "sushi_restaurant": "sushi",
    "ramen_restaurant": "ramen",
    "pizza_restaurant": "pizza",
    "hamburger_restaurant": "burgers",
    "steak_house": "steakhouse",
    "barbecue_restaurant": "bbq",
    "sandwich_shop": "sandwiches",

    # Cafes & Desserts
    "cafe": "cafe",
    "coffee_shop": "coffee",
    "bakery": "bakery",
    "dessert_shop": "desserts",
    "ice_cream_shop": "ice-cream",

    # Other
    "food_court": "hawker-food-court",
    "bar": "drinks",
    "wine_bar": "wine"
}

# Attraction type mappings - Extract attraction-specific tags from Google Place types
ATTRACTION_TYPE_MAPPINGS = {
    # Museums & Culture
    "museum": "museum",
    "art_gallery": "art",
    "historical_landmark": "historical",
    "cultural_landmark": "cultural",
    "monument": "landmark",

    # Entertainment
    "amusement_park": "theme-park",
    "amusement_center": "entertainment",
    "zoo": "zoo",
    "aquarium": "aquarium",
    "water_park": "water-park",
    "movie_theater": "cinema",
    "performing_arts_theater": "theater",

    # Nature & Outdoors
    "park": "park",
    "national_park": "nature",
    "botanical_garden": "gardens",
    "beach": "beach",
    "hiking_area": "hiking",

    # Religious
    "church": "religious",
    "hindu_temple": "temple",
    "mosque": "mosque",
    "synagogue": "religious",

    # Shopping
    "shopping_mall": "shopping",
    "market": "market",

    # Sports & Recreation
    "sports_complex": "sports",
    "stadium": "sports",
    "swimming_pool": "swimming",
    "gym": "fitness",

    # Wellness
    "spa": "spa",
    "wellness_center": "wellness",

    # Tourist
    "tourist_attraction": "tourist-spot",
    "observation_deck": "viewpoint",

    # Transport
    "airport": "airport",
    "international_airport": "airport"
}

# Singapore area mappings - Tag places by neighborhood/area
SINGAPORE_AREA_MAPPINGS = {
    # Central
    "marina bay": "marina-bay",
    "cbd": "cbd",
    "raffles place": "cbd",
    "city hall": "city-hall",
    "orchard": "orchard",
    "orchard road": "orchard",

    # Cultural districts
    "chinatown": "chinatown",
    "little india": "little-india",
    "kampong glam": "kampong-glam",
    "arab street": "arab-street",

    # Tourist areas
    "sentosa": "sentosa",
    "clarke quay": "clarke-quay",
    "boat quay": "riverside",
    "robertson quay": "riverside",

    # Residential/Heartlands
    "tiong bahru": "tiong-bahru",
    "katong": "katong",
    "joo chiat": "joo-chiat",
    "holland village": "holland-village",
    "dempsey": "dempsey",

    # East
    "changi": "changi",
    "east coast": "east-coast",
    "bedok": "heartland",
    "tampines": "heartland",

    # West
    "jurong": "jurong",
    "bukit batok": "heartland",
    "clementi": "heartland",

    # North
    "woodlands": "heartland",
    "yishun": "heartland",
    "ang mo kio": "heartland",

    # Central-North
    "bishan": "bishan",
    "toa payoh": "heartland",
    "novena": "novena",
    "bugis": "bugis"
}

# ============================================================================
# SINGAPORE GEO CLUSTERS - Geographic search distribution
# ============================================================================
# Region boundaries (for location -> region mapping)
SINGAPORE_GEOCLUSTERS_BOUNDARIES = {
    "north": {"lat_min": 1.410, "lat_max": 1.470, "lon_min": 103.750, "lon_max": 103.860},
    "northeast": {"lat_min": 1.340, "lat_max": 1.425, "lon_min": 103.850, "lon_max": 103.925},
    "east": {"lat_min": 1.280, "lat_max": 1.380, "lon_min": 103.900, "lon_max": 103.985},
    "west": {"lat_min": 1.295, "lat_max": 1.410, "lon_min": 103.685, "lon_max": 103.775},
    "central": {"lat_min": 1.315, "lat_max": 1.390, "lon_min": 103.820, "lon_max": 103.870},
    "downtown": {"lat_min": 1.260, "lat_max": 1.320, "lon_min": 103.825, "lon_max": 103.885},
    "south": {"lat_min": 1.225, "lat_max": 1.315, "lon_min": 103.775, "lon_max": 103.835},
}


# Geographic clusters with multiple search points for food search distribution across Singapore
SINGAPORE_GEOCLUSTERS_POINTS = {
    "north": {"lat": 1.4370, "lon": 103.7865}, # Woodlands Town Centre
    "northeast": {"lat": 1.4041, "lon": 103.9025}, # Punggol Central
    "east": {"lat": 1.3236, "lon": 103.9273}, # Bedok Town Centre
    "central": {"lat": 1.3514, "lon": 103.8487}, # Bishan Central
    "downtown": {"lat": 1.2838, "lon": 103.8607}, # Marina Bay/CBD
    "west": {"lat": 1.3329, "lon": 103.7436}, # Jurong East Hub
    "south": {"lat": 1.2654, "lon": 103.8218}, # HarbourFront/Sentosa Gateway
}

# ============================================================================
# PLACE MULTIPLIERS - Calculate required number of places
# ============================================================================

# Minimum multiplier for attractions (pace * days * multiplier)
ATTRACTION_MULTIPLIER = 2.0

# Multiplier for food (geo clusters * days * multiplier)
FOOD_MULTIPLIER = 1.0

# ============================================================================
# FOOD SEARCH PARAMETERS BY GEO CLUSTER
# ============================================================================
# Different clusters have different rating thresholds and search radius

FOOD_SEARCH_PARAMS_BY_CLUSTER = {
    "north": {
        "min_rating": 4.3,
        "initial_radius": 2500  # meters
    },
    "northeast": {
        "min_rating": 4.3,
        "initial_radius": 2500  # meters
    },
    # All other clusters use default parameters (defined in search_with_requirements)
    # Default: min_rating=4.5, initial_radius=5000
}
