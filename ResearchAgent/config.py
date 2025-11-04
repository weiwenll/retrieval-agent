"""
Comprehensive Interest and Uninterest Mapping Configuration
Maps user preferences (both positive and negative) to Google Places API v1 types
"""

# ============================================================================
# INTEREST MAPPINGS - What users are looking for
# ============================================================================

INTEREST_MAPPINGS = {
    # ========== FOOD & DINING ==========
    # General Food
    "food": ["restaurant", "cafe", "bakery", "food_court"],
    "dining": ["restaurant", "fine_dining_restaurant"],
    "eating": ["restaurant", "fast_food_restaurant", "food_court"],
    "meals": ["restaurant", "meal_takeaway", "meal_delivery"],
    "cuisine": ["restaurant"],
    "foodie": ["restaurant", "fine_dining_restaurant", "food_court"],
    "gastronomy": ["fine_dining_restaurant", "restaurant"],
    
    # Specific Cuisines
    "asian": ["asian_restaurant", "chinese_restaurant", "japanese_restaurant", "korean_restaurant", "thai_restaurant", "vietnamese_restaurant", "indonesian_restaurant"],
    "chinese": ["chinese_restaurant"],
    "japanese": ["japanese_restaurant", "sushi_restaurant", "ramen_restaurant"],
    "korean": ["korean_restaurant"],
    "thai": ["thai_restaurant"],
    "vietnamese": ["vietnamese_restaurant"],
    "indian": ["indian_restaurant"],
    "indonesian": ["indonesian_restaurant"],
    "italian": ["italian_restaurant", "pizza_restaurant"],
    "french": ["french_restaurant"],
    "mexican": ["mexican_restaurant"],
    "spanish": ["spanish_restaurant"],
    "greek": ["greek_restaurant"],
    "mediterranean": ["mediterranean_restaurant", "greek_restaurant", "lebanese_restaurant"],
    "middle eastern": ["middle_eastern_restaurant", "lebanese_restaurant", "turkish_restaurant"],
    "american": ["american_restaurant", "hamburger_restaurant", "steak_house"],
    "brazilian": ["brazilian_restaurant"],
    "african": ["african_restaurant"],
    "afghani": ["afghani_restaurant"],
    
    # Meal Types
    "breakfast": ["breakfast_restaurant", "brunch_restaurant", "cafe", "bakery"],
    "brunch": ["brunch_restaurant", "breakfast_restaurant", "cafe"],
    "lunch": ["restaurant", "fast_food_restaurant", "sandwich_shop", "deli"],
    "dinner": ["restaurant", "fine_dining_restaurant", "steak_house"],
    "buffet": ["buffet_restaurant"],
    
    # Dietary Preferences
    "vegetarian": ["vegetarian_restaurant", "vegan_restaurant"],
    "vegan": ["vegan_restaurant", "vegetarian_restaurant"],
    "seafood": ["seafood_restaurant", "sushi_restaurant"],
    "meat": ["steak_house", "barbecue_restaurant", "hamburger_restaurant"],
    "bbq": ["barbecue_restaurant", "bar_and_grill"],
    "barbecue": ["barbecue_restaurant", "bar_and_grill"],
    "steak": ["steak_house"],
    "sushi": ["sushi_restaurant", "japanese_restaurant"],
    "ramen": ["ramen_restaurant", "japanese_restaurant"],
    "pizza": ["pizza_restaurant", "italian_restaurant"],
    "burger": ["hamburger_restaurant", "fast_food_restaurant"],
    "sandwich": ["sandwich_shop", "deli"],
    
    # Cafes & Desserts
    "cafe": ["cafe", "coffee_shop", "tea_house", "cat_cafe", "dog_cafe"],
    "coffee": ["coffee_shop", "cafe"],
    "tea": ["tea_house", "cafe"],
    "dessert": ["dessert_shop", "dessert_restaurant", "ice_cream_shop", "bakery"],
    "ice cream": ["ice_cream_shop"],
    "bakery": ["bakery"],
    "pastry": ["bakery", "confectionery"],
    "chocolate": ["chocolate_shop", "chocolate_factory", "confectionery"],
    "sweets": ["candy_store", "confectionery", "dessert_shop"],
    "juice": ["juice_shop"],
    "smoothie": ["juice_shop", "acai_shop"],
    "acai": ["acai_shop"],
    
    # Quick Bites
    "fast food": ["fast_food_restaurant"],
    "quick": ["fast_food_restaurant", "meal_takeaway", "food_court"],
    "takeaway": ["meal_takeaway", "fast_food_restaurant"],
    "delivery": ["meal_delivery"],
    "street food": ["food_court", "market"],
    "snack": ["cafe", "bakery", "convenience_store"],
    "deli": ["deli", "sandwich_shop"],
    
    # Bars & Nightlife
    "bar": ["bar", "pub", "wine_bar", "bar_and_grill"],
    "pub": ["pub", "bar"],
    "drinks": ["bar", "pub", "wine_bar"],
    "cocktails": ["bar", "wine_bar"],
    "wine": ["wine_bar", "bar"],
    "beer": ["pub", "bar"],
    "nightlife": ["night_club", "bar", "pub", "casino"],
    "club": ["night_club"],
    "nightclub": ["night_club"],
    "party": ["night_club", "bar"],
    
    # ========== ATTRACTIONS & ENTERTAINMENT ==========
    # Museums & Culture
    "museum": ["museum", "art_gallery"],
    "art": ["art_gallery", "art_studio", "museum"],
    "gallery": ["art_gallery"],
    "culture": ["museum", "cultural_center", "cultural_landmark", "performing_arts_theater"],
    "cultural": ["cultural_center", "cultural_landmark", "museum"],
    "history": ["museum", "historical_landmark", "historical_place", "monument"],
    "historical": ["historical_landmark", "historical_place", "museum"],
    "heritage": ["museum", "historical_landmark", "monument"],
    "monument": ["monument", "historical_landmark"],
    "sculpture": ["sculpture", "art_gallery"],
    
    # Entertainment
    "entertainment": ["amusement_park", "amusement_center", "movie_theater", "performing_arts_theater"],
    "theater": ["movie_theater", "performing_arts_theater", "concert_hall"],
    "theatre": ["movie_theater", "performing_arts_theater", "concert_hall"],
    "movie": ["movie_theater"],
    "cinema": ["movie_theater"],
    "concert": ["concert_hall", "amphitheatre", "event_venue"],
    "music": ["concert_hall", "philharmonic_hall", "night_club"],
    "comedy": ["comedy_club"],
    "show": ["performing_arts_theater", "concert_hall", "event_venue"],
    "opera": ["opera_house", "performing_arts_theater"],
    "dance": ["dance_hall", "night_club"],
    "karaoke": ["karaoke"],
    
    # Family & Kids
    "family": ["amusement_park", "zoo", "aquarium", "park", "playground"],
    "kids": ["amusement_park", "zoo", "aquarium", "playground", "childrens_camp"],
    "children": ["playground", "amusement_park", "zoo", "aquarium", "childrens_camp"],
    "playground": ["playground", "park"],
    
    # Theme Parks & Attractions
    "theme park": ["amusement_park"],
    "amusement": ["amusement_park", "amusement_center"],
    "roller coaster": ["roller_coaster", "amusement_park"],
    "ferris wheel": ["ferris_wheel", "amusement_park"],
    "water park": ["water_park"],
    "zoo": ["zoo", "wildlife_park"],
    "aquarium": ["aquarium"],
    "animals": ["zoo", "aquarium", "wildlife_park", "wildlife_refuge"],
    "wildlife": ["wildlife_park", "wildlife_refuge", "zoo"],
    
    # Gaming & Recreation
    "gaming": ["video_arcade", "amusement_center", "casino"],
    "arcade": ["video_arcade", "amusement_center"],
    "bowling": ["bowling_alley"],
    "casino": ["casino"],
    "gambling": ["casino"],
    
    # ========== OUTDOOR & NATURE ==========
    "outdoor": ["park", "hiking_area", "beach", "camping_cabin", "campground"],
    "nature": ["park", "national_park", "state_park", "botanical_garden", "hiking_area"],
    "park": ["park", "national_park", "state_park", "dog_park"],
    "garden": ["botanical_garden", "garden"],
    "botanical": ["botanical_garden"],
    "beach": ["beach"],
    "hiking": ["hiking_area", "park", "national_park"],
    "camping": ["campground", "camping_cabin", "rv_park"],
    "picnic": ["picnic_ground", "park", "barbecue_area"],
    "bbq area": ["barbecue_area", "picnic_ground"],
    "scenic": ["observation_deck", "park", "botanical_garden"],
    "viewpoint": ["observation_deck"],
    "adventure": ["adventure_sports_center", "hiking_area", "off_roading_area"],
    "cycling": ["cycling_park", "park"],
    "skateboard": ["skateboard_park"],
    "marina": ["marina"],
    
    # ========== SPORTS & FITNESS ==========
    "sports": ["sports_complex", "stadium", "arena", "athletic_field"],
    "fitness": ["gym", "fitness_center", "sports_complex"],
    "gym": ["gym", "fitness_center"],
    "workout": ["gym", "fitness_center"],
    "swimming": ["swimming_pool"],
    "pool": ["swimming_pool"],
    "golf": ["golf_course"],
    "tennis": ["sports_complex", "athletic_field"],
    "basketball": ["sports_complex", "athletic_field"],
    "football": ["stadium", "athletic_field"],
    "soccer": ["stadium", "athletic_field"],
    "stadium": ["stadium", "arena"],
    "ice skating": ["ice_skating_rink"],
    "skiing": ["ski_resort"],
    "fishing": ["fishing_pond", "fishing_charter"],
    
    # ========== WELLNESS & RELAXATION ==========
    "spa": ["spa", "wellness_center"],
    "wellness": ["wellness_center", "spa", "yoga_studio"],
    "massage": ["massage", "spa"],
    "yoga": ["yoga_studio", "wellness_center"],
    "meditation": ["yoga_studio", "wellness_center"],
    "relax": ["spa", "park", "beach"],
    "relaxation": ["spa", "wellness_center", "park"],
    "sauna": ["sauna", "spa"],
    "beauty": ["beauty_salon", "beautician", "spa"],
    "salon": ["beauty_salon", "hair_salon"],
    "hair": ["hair_salon", "hair_care", "barber_shop"],
    "nails": ["nail_salon", "beauty_salon"],
    "tanning": ["tanning_studio"],
    "skin care": ["skin_care_clinic", "spa"],
    
    # ========== SHOPPING ==========
    "shopping": ["shopping_mall", "department_store", "clothing_store"],
    "mall": ["shopping_mall"],
    "shops": ["shopping_mall", "store"],
    "boutique": ["clothing_store", "gift_shop"],
    "market": ["market", "grocery_store"],
    "clothes": ["clothing_store", "department_store"],
    "fashion": ["clothing_store", "department_store", "shoe_store"],
    "shoes": ["shoe_store"],
    "jewelry": ["jewelry_store"],
    "electronics": ["electronics_store"],
    "books": ["book_store", "library"],
    "bookstore": ["book_store"],
    "gifts": ["gift_shop"],
    "souvenirs": ["gift_shop", "store"],
    "furniture": ["furniture_store", "home_goods_store"],
    "home decor": ["home_goods_store", "furniture_store"],
    "groceries": ["grocery_store", "supermarket"],
    "supermarket": ["supermarket", "grocery_store"],
    "convenience": ["convenience_store"],
    "pharmacy": ["pharmacy", "drugstore"],
    "liquor": ["liquor_store"],
    "pet": ["pet_store"],
    "sports equipment": ["sporting_goods_store"],
    "hardware": ["hardware_store", "home_improvement_store"],
    
    # ========== ACCOMMODATION ==========
    "hotel": ["hotel", "resort_hotel"],
    "accommodation": ["lodging", "hotel"],
    "stay": ["lodging", "hotel"],
    "lodging": ["lodging"],
    "resort": ["resort_hotel"],
    "hostel": ["hostel"],
    "motel": ["motel"],
    "inn": ["inn", "japanese_inn", "budget_japanese_inn"],
    "bed and breakfast": ["bed_and_breakfast"],
    "guesthouse": ["guest_house"],
    "vacation rental": ["cottage", "private_guest_room"],
    "camping": ["campground", "camping_cabin", "rv_park"],
    "farmstay": ["farmstay"],
    
    # ========== RELIGIOUS & SPIRITUAL ==========
    "temple": ["hindu_temple", "buddhist_temple"],
    "church": ["church"],
    "mosque": ["mosque"],
    "synagogue": ["synagogue"],
    "worship": ["church", "hindu_temple", "mosque", "synagogue"],
    "religious": ["church", "hindu_temple", "mosque", "synagogue"],
    "spiritual": ["church", "hindu_temple", "mosque", "synagogue"],
    "prayer": ["church", "hindu_temple", "mosque", "synagogue"],
    
    # ========== EDUCATION & LEARNING ==========
    "library": ["library"],
    "education": ["museum", "library", "university"],
    "learning": ["museum", "library", "cultural_center"],
    "school": ["school", "primary_school", "secondary_school", "university"],
    "university": ["university"],
    "college": ["university"],
    "preschool": ["preschool"],
    
    # ========== TRANSPORTATION ==========
    "transport": ["train_station", "bus_station", "subway_station"],
    "transportation": ["transit_station", "transit_depot"],
    "train": ["train_station"],
    "bus": ["bus_station", "bus_stop"],
    "subway": ["subway_station"],
    "metro": ["subway_station"],
    "airport": ["airport", "international_airport"],
    "ferry": ["ferry_terminal"],
    "taxi": ["taxi_stand"],
    
    # ========== TOURIST SPECIFIC ==========
    "tourist": ["tourist_attraction", "tourist_information_center"],
    "attraction": ["tourist_attraction"],
    "sightseeing": ["tourist_attraction", "observation_deck"],
    "landmark": ["historical_landmark", "cultural_landmark", "monument"],
    "iconic": ["tourist_attraction", "monument"],
    "must see": ["tourist_attraction", "historical_landmark"],
    "photo spot": ["observation_deck", "tourist_attraction"],
    "instagram": ["tourist_attraction", "observation_deck"],
    "visitor center": ["visitor_center", "tourist_information_center"],
    "tour": ["tour_agency", "travel_agency"],
}

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
# TAG GENERATION MAPPINGS - Convert Google Place types to descriptive tags
# ============================================================================

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
