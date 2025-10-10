"""
Configuration for Research Agent
Contains all mappings, constants, and configuration settings
"""

# Common interest mappings dictionary (covers ~80% of typical user interests)
COMMON_INTEREST_MAPPINGS = {
    # Museums and cultural
    "museums": "museum", "museum": "museum", "cultural": "museum", "culture": "museum",
    "art": "museum", "arts": "museum", "gallery": "museum", "galleries": "museum",
    "history": "museum", "historical": "museum", "heritage": "museum", "educational": "museum",

    # Parks and nature
    "parks": "park", "park": "park", "gardens": "park", "garden": "park",
    "nature": "park", "outdoor": "park", "outdoors": "park", "green spaces": "park",
    "botanical": "park", "recreation": "park",

    # Food and dining
    "food": "food", "foods": "food", "dining": "food", "restaurants": "food",
    "restaurant": "food", "cuisine": "food",

    # Cafes and beverages
    "cafe": "cafe", "cafes": "cafe", "coffee": "cafe", "tea": "cafe", "dessert": "cafe",
    "desserts": "cafe", "pastry": "bakery", "pastries": "bakery", "bakery": "bakery",

    # Bars and nightlife
    "bar": "bar", "bars": "bar", "pub": "bar", "pubs": "bar", "drinks": "bar",
    "nightlife": "bar", "cocktails": "bar", "beer": "bar", "wine": "bar",

    # Shopping
    "shopping": "shopping_mall", "shops": "shopping_mall", "mall": "shopping_mall",
    "malls": "shopping_mall", "retail": "shopping_mall", "boutiques": "shopping_mall",
    "markets": "shopping_mall", "market": "shopping_mall",

    # Tourist attractions
    "attractions": "tourist_attraction", "attraction": "tourist_attraction",
    "sightseeing": "tourist_attraction", "landmarks": "tourist_attraction",
    "landmark": "tourist_attraction", "tourist": "tourist_attraction",
    "family": "tourist_attraction", "kids": "tourist_attraction", "children": "tourist_attraction",
    "entertainment": "tourist_attraction", "fun": "tourist_attraction",

    # Accommodation
    "hotel": "lodging", "hotels": "lodging", "accommodation": "lodging",
    "stay": "lodging", "lodging": "lodging", "hostel": "lodging", "motel": "lodging"
}

# Dietary keywords that should append " food"
DIETARY_KEYWORDS = {
    "vegetarian", "vegan", "halal", "kosher", "gluten-free",
    "gluten free", "organic", "plant-based", "local cuisine", "local"
}

# Food-specific type mappings for tag generation
FOOD_TYPE_MAPPINGS = {
    "chinese_restaurant": "chinese",
    "japanese_restaurant": "japanese",
    "indian_restaurant": "indian",
    "italian_restaurant": "italian",
    "french_restaurant": "french",
    "korean_restaurant": "korean",
    "thai_restaurant": "thai",
    "vietnamese_restaurant": "vietnamese",
    "mexican_restaurant": "mexican",
    "mediterranean_restaurant": "mediterranean",
    "seafood_restaurant": "seafood",
    "steakhouse": "steak",
    "vegetarian_restaurant": "vegetarian",
    "vegan_restaurant": "vegan",
    "bakery": "bakery",
    "cafe": "cafe",
    "bar": "bar",
    "meal_takeaway": "takeaway",
    "meal_delivery": "delivery",
    "fast_food_restaurant": "fast-food",
    "breakfast_restaurant": "breakfast",
    "brunch_restaurant": "brunch",
    "fine_dining_restaurant": "fine-dining",
    "hawker_centre": "hawker"
}

# Attraction-specific type mappings for tag generation
ATTRACTION_TYPE_MAPPINGS = {
    "museum": "museum",
    "art_gallery": "art",
    "zoo": "zoo",
    "aquarium": "aquarium",
    "amusement_park": "theme-park",
    "park": "park",
    "national_park": "nature",
    "shopping_mall": "shopping",
    "night_club": "nightlife",
    "tourist_attraction": "attraction",
    "place_of_worship": "cultural",
    "hindu_temple": "hindu",
    "church": "christian",
    "mosque": "islamic",
    "buddhist_temple": "buddhist",
    "historical_landmark": "historical",
    "natural_feature": "nature",
    "campground": "outdoor",
    "hiking_area": "hiking",
    "beach": "beach",
    "spa": "wellness"
}

# Singapore-specific area mappings for location tags
SINGAPORE_AREA_MAPPINGS = {
    "sentosa": "sentosa",
    "marina bay": "marina-bay",
    "orchard": "orchard",
    "chinatown": "chinatown",
    "little india": "little-india",
    "kampong glam": "kampong-glam",
    "clarke quay": "clarke-quay",
    "east coast": "east-coast",
    "gardens by the bay": "gardens-by-the-bay",
    "bugis": "bugis",
    "raffles": "raffles"
}

# Tag extraction prompts for LLM
TAG_EXTRACTION_PROMPT = """Extract 2-3 relevant descriptive tags from this place description for travelers.

Place: {name}
Type: {place_type}
Description: {description}

Existing tags: {existing_tags}

Rules:
- Extract ONLY tags NOT already in existing tags
- Focus on unique features, activities, atmosphere, or characteristics
- Use lowercase with hyphens (e.g., "family-friendly", "photo-spot", "iconic")
- Return ONLY a JSON array of 2-3 new tags
- If no relevant new tags, return empty array []

Examples of good tags:
- For attractions: "iconic", "photo-spot", "family-friendly", "interactive", "educational", "outdoor", "indoor"
- For food: "local-favorite", "authentic", "fusion", "street-food", "upscale", "cozy", "waterfront"
- For parks: "scenic", "relaxing", "jogging", "picnic-spot", "botanical"

Return only JSON array: ["tag1", "tag2"]"""

TAG_EXTRACTION_SYSTEM_PROMPT = "You extract relevant travel tags from place descriptions. Return only JSON array."

# LLM Configuration
LLM_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens_tags": 50
}

# Tag generation limits
TAG_LIMITS = {
    "max_total_tags": 8,
    "max_llm_tags": 3
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
