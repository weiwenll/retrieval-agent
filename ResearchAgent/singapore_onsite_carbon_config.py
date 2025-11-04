
# ===================================================================
# CARBON EMISSION FACTORS (kg CO2e per visit)
# ===================================================================
# Categories: 

PLACE_CARBON_FACTORS = {
    # BUSINESS
    "farm": {
        "co2e_kg": 0.3,
        "low_carbon_score": 15.0,
        "notes": "Minimal infrastructure, mostly outdoor operations, low energy use"
    },
    "ranch": {
        "co2e_kg": 0.4,
        "low_carbon_score": 18.0,
        "notes": "Similar to farms with slightly more equipment and facilities"
    },
    # CULTURE
    "art_gallery": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Climate control for preservation, security systems, specialized lighting"
    },
    "art_studio": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Working space with basic HVAC, lighting for detailed work"
    },
    "auditorium": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Large space HVAC, stage lighting, sound systems"
    },
    "cultural_landmark": {
        "co2e_kg": 0.6,
        "low_carbon_score": 22.0,
        "notes": "Often outdoor or semi-outdoor, minimal climate control"
    },
    "historical_place": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Preserved sites with basic lighting and maintenance"
    },
    "monument": {
        "co2e_kg": 0.2,
        "low_carbon_score": 10.0,
        "notes": "Mostly outdoor, minimal energy use except night lighting"
    },
    "museum": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Strict climate control, preservation systems, security"
    },
    "performing_arts_theater": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Stage lighting, sound systems, HVAC for large audiences"
    },
    "sculpture": {
        "co2e_kg": 0.1,
        "low_carbon_score": 5.0,
        "notes": "Outdoor installation with minimal infrastructure"
    },
    # EDUCATION
    "school": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Classrooms with HVAC, computer labs, cafeteria operations"
    },
    "library": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Climate control for book preservation, computer stations"
    },
    "university": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Multiple facilities, labs, lecture halls, extensive HVAC"
    },
    # ENTERTAINMENT AND RECREATION
    "adventure_sports_center": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Equipment operations, safety systems, facility maintenance"
    },
    "amphitheatre": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Often outdoor, sound and lighting systems for events"
    },
    "amusement_center": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Gaming machines, HVAC, lighting, electronic equipment"
    },
    "amusement_park": {
        "co2e_kg": 8.5,
        "low_carbon_score": 75.0,
        "notes": "Rides, extensive lighting, cooling, high energy footprint"
    },
    "aquarium": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Water filtration, temperature control, life support systems"
    },
    "banquet_hall": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Event HVAC, kitchen operations, lighting systems"
    },
    "barbecue_area": {
        "co2e_kg": 0.5,
        "low_carbon_score": 20.0,
        "notes": "Outdoor area with minimal infrastructure"
    },
    "botanical_garden": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Some climate-controlled sections, irrigation systems"
    },
    "bowling_alley": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Lane machinery, HVAC, specialized lighting"
    },
    "casino": {
        "co2e_kg": 6.0,
        "low_carbon_score": 68.0,
        "notes": "24/7 operations, gaming systems, heavy HVAC and lighting"
    },
    "childrens_camp": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Mostly outdoor activities, basic facilities"
    },
    "comedy_club": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Sound systems, stage lighting, HVAC for audiences"
    },
    "community_center": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Multi-purpose spaces, basic HVAC, community services"
    },
    "concert_hall": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Acoustic requirements, stage systems, audience HVAC"
    },
    "convention_center": {
        "co2e_kg": 5.0,
        "low_carbon_score": 65.0,
        "notes": "Large-scale HVAC, AV equipment, exhibition lighting"
    },
    "cultural_center": {
        "co2e_kg": 1.6,
        "low_carbon_score": 40.0,
        "notes": "Multi-purpose cultural facilities, moderate energy use"
    },
    "cycling_park": {
        "co2e_kg": 0.1,
        "low_carbon_score": 5.0,
        "notes": "Outdoor facility with minimal infrastructure"
    },
    "dance_hall": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Sound systems, specialized flooring, HVAC"
    },
    "dog_park": {
        "co2e_kg": 0.0,
        "low_carbon_score": 2.0,
        "notes": "Outdoor area with minimal facilities"
    },
    "event_venue": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Variable based on event type, AV equipment, HVAC"
    },
    "ferris_wheel": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Mechanical operation, lighting, safety systems"
    },
    "garden": {
        "co2e_kg": 0.2,
        "low_carbon_score": 10.0,
        "notes": "Irrigation, minimal lighting, maintenance equipment"
    },
    "hiking_area": {
        "co2e_kg": 0.0,
        "low_carbon_score": 0.0,
        "notes": "Natural area with trail maintenance only"
    },
    "historical_landmark": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Preservation lighting, visitor facilities"
    },
    "internet_cafe": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Multiple computers, HVAC, networking equipment"
    },
    "karaoke": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Sound systems, private room HVAC, AV equipment"
    },
    "marina": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Dock operations, fuel storage, marine facilities"
    },
    "movie_rental": {
        "co2e_kg": 0.5,
        "low_carbon_score": 20.0,
        "notes": "Small retail operation, minimal energy use"
    },
    "movie_theater": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Projection systems, sound, HVAC for audiences"
    },
    "national_park": {
        "co2e_kg": 0.01,
        "low_carbon_score": 1.0,
        "notes": "Protected natural area with visitor centers"
    },
    "night_club": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Sound/lighting systems, late-night HVAC operations"
    },
    "observation_deck": {
        "co2e_kg": 1.7,
        "low_carbon_score": 41.0,
        "notes": "Elevator systems, viewing area HVAC, lighting"
    },
    "off_roading_area": {
        "co2e_kg": 0.2,
        "low_carbon_score": 10.0,
        "notes": "Outdoor area with minimal facilities"
    },
    "opera_house": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Acoustic requirements, elaborate stage systems"
    },
    "park": {
        "co2e_kg": 0.0,
        "low_carbon_score": 0.0,
        "notes": "Natural space with basic maintenance"
    },
    "philharmonic_hall": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Acoustic design, climate control for instruments"
    },
    "picnic_ground": {
        "co2e_kg": 0.0,
        "low_carbon_score": 0.0,
        "notes": "Outdoor area with minimal infrastructure"
    },
    "planetarium": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Projection systems, dome HVAC, specialized equipment"
    },
    "plaza": {
        "co2e_kg": 0.1,
        "low_carbon_score": 5.0,
        "notes": "Open public space with minimal energy use"
    },
    "roller_coaster": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Mechanical systems, safety equipment, operations"
    },
    "skateboard_park": {
        "co2e_kg": 0.1,
        "low_carbon_score": 5.0,
        "notes": "Outdoor facility with lighting"
    },
    "state_park": {
        "co2e_kg": 0.01,
        "low_carbon_score": 1.0,
        "notes": "Natural area with basic visitor facilities"
    },
    "tourist_attraction": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Variable facilities and energy use"
    },
    "point_of_interest": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "General tourist site, variable facilities, visitor amenities"
    },
    "video_arcade": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Gaming machines, HVAC, lighting, electronics"
    },
    "visitor_center": {
        "co2e_kg": 0.9,
        "low_carbon_score": 30.0,
        "notes": "Information displays, basic HVAC, lighting"
    },
    "water_park": {
        "co2e_kg": 6.0,
        "low_carbon_score": 68.0,
        "notes": "Water pumps, filtration, slides, extensive operations"
    },
    "wedding_venue": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Event lighting, HVAC, catering facilities"
    },
    "wildlife_park": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Animal facilities, visitor amenities"
    },
    "wildlife_refuge": {
        "co2e_kg": 0.4,
        "low_carbon_score": 18.0,
        "notes": "Protected area with minimal infrastructure"
    },
    "zoo": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Animal habitats, climate control, visitor facilities"
    },
    # FACILITIES
    "public_bath": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Water heating, filtration, HVAC, high water use"
    },
    "public_bathroom": {
        "co2e_kg": 0.2,
        "low_carbon_score": 10.0,
        "notes": "Basic facilities, water systems, lighting"
    },
    "stable": {
        "co2e_kg": 0.6,
        "low_carbon_score": 22.0,
        "notes": "Animal care, basic ventilation, lighting"
    },
    # FOOD AND DRINK
    "acai_shop": {
        "co2e_kg": 1.4,
        "low_carbon_score": 37.0,
        "notes": "Refrigeration, blenders, HVAC"
    },
    "afghani_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Kitchen operations, traditional cooking methods"
    },
    "african_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Grilling operations, kitchen equipment, HVAC"
    },
    "american_restaurant": {
        "co2e_kg": 2.6,
        "low_carbon_score": 51.0,
        "notes": "Grills, fryers, extensive kitchen operations"
    },
    "asian_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Wok cooking, rice cookers, refrigeration"
    },
    "bagel_shop": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Ovens, refrigeration, display cases"
    },
    "bakery": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Multiple ovens, mixers, refrigeration, display"
    },
    "bar": {
        "co2e_kg": 1.6,
        "low_carbon_score": 40.0,
        "notes": "Refrigeration, ice machines, sound systems"
    },
    "bar_and_grill": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Grills, refrigeration, bar equipment"
    },
    "barbecue_restaurant": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Smokers, grills, high-heat cooking"
    },
    "brazilian_restaurant": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Churrasco grills, continuous cooking operations"
    },
    "breakfast_restaurant": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Griddles, coffee machines, morning operations"
    },
    "brunch_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Extended cooking hours, diverse menu equipment"
    },
    "buffet_restaurant": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Food warmers, extensive displays, high waste"
    },
    "cafe": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Coffee machines, light cooking, refrigeration"
    },
    "cafeteria": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Bulk cooking, warming stations, high volume"
    },
    "candy_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Display refrigeration, HVAC, lighting"
    },
    "cat_cafe": {
        "co2e_kg": 1.3,
        "low_carbon_score": 36.0,
        "notes": "Coffee operations plus animal area HVAC"
    },
    "chinese_restaurant": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "High-heat wok cooking, steamers, refrigeration"
    },
    "chocolate_factory": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Temperature control, production equipment"
    },
    "chocolate_shop": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Climate control for chocolate, display refrigeration"
    },
    "coffee_shop": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Espresso machines, grinders, refrigeration"
    },
    "confectionery": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Cooking equipment, temperature control, display"
    },
    "deli": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Slicers, refrigeration cases, sandwich prep"
    },
    "dessert_restaurant": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Refrigeration, ovens for baking, display freezers"
    },
    "dessert_shop": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Freezers, refrigerated displays, HVAC"
    },
    "diner": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Griddles, fryers, 24-hour operations common"
    },
    "dog_cafe": {
        "co2e_kg": 1.3,
        "low_carbon_score": 36.0,
        "notes": "Coffee operations plus pet area ventilation"
    },
    "donut_shop": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Fryers, proofing equipment, early morning operations"
    },
    "fast_food_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "High-volume fryers, grills, warming stations"
    },
    "fine_dining_restaurant": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Sophisticated kitchen equipment, enhanced HVAC"
    },
    "food_court": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Multiple vendors, shared HVAC, varied cooking"
    },
    "french_restaurant": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Complex cooking techniques, wine storage"
    },
    "greek_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Grills, ovens, Mediterranean cooking methods"
    },
    "hamburger_restaurant": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Grills, fryers, high-volume operations"
    },
    "ice_cream_shop": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Heavy freezer use, refrigerated displays"
    },
    "indian_restaurant": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Tandoor ovens, extended cooking times"
    },
    "indonesian_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Wok cooking, grilling, rice cookers"
    },
    "italian_restaurant": {
        "co2e_kg": 2.6,
        "low_carbon_score": 51.0,
        "notes": "Pizza ovens, pasta cooking, wine storage"
    },
    "japanese_restaurant": {
        "co2e_kg": 2.3,
        "low_carbon_score": 47.0,
        "notes": "Sushi refrigeration, grills, rice cookers"
    },
    "juice_shop": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Refrigeration, juicers, minimal cooking"
    },
    "korean_restaurant": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Table grills, kimchi refrigeration, ventilation"
    },
    "lebanese_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Grills, ovens, Middle Eastern cooking"
    },
    "meal_delivery": {
        "co2e_kg": 3.2,
        "low_carbon_score": 56.0,
        "notes": "Kitchen operations plus delivery emissions"
    },
    "meal_takeaway": {
        "co2e_kg": 2.6,
        "low_carbon_score": 51.0,
        "notes": "Kitchen operations plus packaging waste"
    },
    "mediterranean_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Grills, ovens, varied cooking methods"
    },
    "mexican_restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "Grills, fryers, tortilla equipment"
    },
    "middle_eastern_restaurant": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Grills, ovens, traditional cooking methods"
    },
    "pizza_restaurant": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "High-temperature pizza ovens, refrigeration"
    },
    "pub": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Kitchen operations, beer refrigeration, HVAC"
    },
    "ramen_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Continuous broth cooking, noodle preparation"
    },
    "restaurant": {
        "co2e_kg": 2.4,
        "low_carbon_score": 48.0,
        "notes": "General kitchen operations, HVAC, refrigeration"
    },
    "sandwich_shop": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Refrigeration, toasters, minimal cooking"
    },
    "seafood_restaurant": {
        "co2e_kg": 2.6,
        "low_carbon_score": 51.0,
        "notes": "Live tanks, extensive refrigeration, cooking"
    },
    "spanish_restaurant": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Paella burners, grills, tapas preparation"
    },
    "steak_house": {
        "co2e_kg": 3.2,
        "low_carbon_score": 56.0,
        "notes": "High-temperature grills, aging refrigeration"
    },
    "sushi_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Extensive refrigeration for raw fish, rice cookers"
    },
    "tea_house": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Hot water systems, minimal cooking, HVAC"
    },
    "thai_restaurant": {
        "co2e_kg": 2.3,
        "low_carbon_score": 47.0,
        "notes": "Wok cooking, rice cookers, refrigeration"
    },
    "turkish_restaurant": {
        "co2e_kg": 2.6,
        "low_carbon_score": 51.0,
        "notes": "Kebab grills, ovens, traditional cooking"
    },
    "vegan_restaurant": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Lower energy cooking, plant-based operations"
    },
    "vegetarian_restaurant": {
        "co2e_kg": 1.9,
        "low_carbon_score": 43.0,
        "notes": "Moderate cooking energy, no meat refrigeration"
    },
    "vietnamese_restaurant": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Pho cooking, wok operations, refrigeration"
    },
    "wine_bar": {
        "co2e_kg": 1.4,
        "low_carbon_score": 37.0,
        "notes": "Wine storage climate control, minimal cooking"
    },
    # GOVERNMENT
    "city_hall": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Office operations, public services, HVAC"
    },
    "courthouse": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Security systems, courtroom HVAC, offices"
    },
    "embassy": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "High security, 24/7 operations, diplomatic facilities"
    },
    "fire_station": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "24/7 operations, equipment maintenance, vehicle bays"
    },
    "government_office": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Standard office operations, HVAC, computers"
    },
    "local_government_office": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Smaller scale office operations, public services"
    },
    "police": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "24/7 operations, detention facilities, dispatch"
    },
    "post_office": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Sorting equipment, retail operations, HVAC"
    },
    # HEALTH AND WELLNESS
    "massage": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Treatment room HVAC, laundry, ambient conditions"
    },
    "sauna": {
        "co2e_kg": 4.0,
        "low_carbon_score": 60.0,
        "notes": "High heat generation, steam systems, ventilation"
    },
    "spa": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Water heating, laundry, treatment rooms, pools, HVAC"
    },
    "wellness_center": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Multiple treatment rooms, pools, HVAC"
    },
    "yoga_studio": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Room heating for hot yoga, basic HVAC"
    },
    # LODGING
    "hotel": {
        "co2e_kg": 12.0,
        "low_carbon_score": 80.0,
        "notes": "Per night stay: 24/7 HVAC, laundry, pools, restaurants"
    },
    "resort_hotel": {
        "co2e_kg": 18.0,
        "low_carbon_score": 85.0,
        "notes": "Per night: Luxury facilities, multiple pools, spas, casinos"
    },
    # NATURAL FEATURES
    "beach": {
        "co2e_kg": 0.0,
        "low_carbon_score": 0.0,
        "notes": "Natural area with minimal facilities"
    },
    # PLACES OF WORSHIP
    "church": {
        "co2e_kg": 0.9,
        "low_carbon_score": 30.0,
        "notes": "Periodic use, HVAC during services, lighting"
    },
    "hindu_temple": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Lighting, incense, prayer operations"
    },
    "mosque": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Ablution facilities, prayer hall HVAC"
    },
    "synagogue": {
        "co2e_kg": 0.9,
        "low_carbon_score": 30.0,
        "notes": "HVAC during services, lighting systems"
    },
    "place_of_worship": {
        "co2e_kg": 0.9,
        "low_carbon_score": 30.0,
        "notes": "General religious facility, periodic use, HVAC during services"
    },
    # SHOPPING
    "asian_grocery_store": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Extensive refrigeration, fresh produce sections"
    },
    "auto_parts_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Warehouse lighting, basic HVAC"
    },
    "bicycle_store": {
        "co2e_kg": 0.6,
        "low_carbon_score": 22.0,
        "notes": "Retail space, workshop area, basic HVAC"
    },
    "book_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "HVAC, lighting, minimal refrigeration"
    },
    "butcher_shop": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Heavy refrigeration, meat processing equipment"
    },
    "cell_phone_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Display devices, charging stations, HVAC"
    },
    "clothing_store": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "HVAC, lighting, fitting rooms"
    },
    "convenience_store": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "24/7 operations, refrigeration, lighting"
    },
    "department_store": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Large HVAC systems, escalators, extensive lighting"
    },
    "discount_store": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Warehouse-style operations, basic HVAC"
    },
    "electronics_store": {
        "co2e_kg": 1.3,
        "low_carbon_score": 36.0,
        "notes": "Display units, demo equipment, HVAC"
    },
    "food_store": {
        "co2e_kg": 2.2,
        "low_carbon_score": 47.0,
        "notes": "Refrigeration, freezers, HVAC"
    },
    "furniture_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Showroom lighting, warehouse operations"
    },
    "gift_shop": {
        "co2e_kg": 0.6,
        "low_carbon_score": 22.0,
        "notes": "Small retail space, basic HVAC, lighting"
    },
    "grocery_store": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Extensive refrigeration, produce sections, bakery"
    },
    "hardware_store": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Warehouse operations, lumber yard, HVAC"
    },
    "home_goods_store": {
        "co2e_kg": 0.9,
        "low_carbon_score": 30.0,
        "notes": "Showroom lighting, HVAC, minimal refrigeration"
    },
    "home_improvement_store": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Large warehouse, outdoor garden center"
    },
    "jewelry_store": {
        "co2e_kg": 1.1,
        "low_carbon_score": 34.0,
        "notes": "Security systems, display lighting, HVAC"
    },
    "liquor_store": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Climate control for wine, refrigeration"
    },
    "market": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "Variable - wet markets to covered markets"
    },
    "pet_store": {
        "co2e_kg": 1.8,
        "low_carbon_score": 42.0,
        "notes": "Animal habitat climate control, aquariums"
    },
    "shoe_store": {
        "co2e_kg": 0.8,
        "low_carbon_score": 28.0,
        "notes": "Retail space HVAC, storage, lighting"
    },
    "shopping_mall": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Large HVAC systems, escalators, common areas"
    },
    "sporting_goods_store": {
        "co2e_kg": 1.0,
        "low_carbon_score": 32.0,
        "notes": "Large retail space, equipment displays"
    },
    "store": {
        "co2e_kg": 1.2,
        "low_carbon_score": 35.0,
        "notes": "General retail operations, HVAC, lighting"
    },
    "supermarket": {
        "co2e_kg": 3.5,
        "low_carbon_score": 58.0,
        "notes": "Heavy refrigeration, freezers, deli operations"
    },
    "warehouse_store": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Large-scale operations, forklift charging"
    },
    "wholesaler": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Warehouse operations, logistics systems"
    },
    # SPORTS
    "arena": {
        "co2e_kg": 5.0,
        "low_carbon_score": 65.0,
        "notes": "Large venue HVAC, lighting, event operations"
    },
    "athletic_field": {
        "co2e_kg": 0.3,
        "low_carbon_score": 15.0,
        "notes": "Outdoor facility, field lighting, minimal buildings"
    },
    "fishing_charter": {
        "co2e_kg": 8.0,
        "low_carbon_score": 72.0,
        "notes": "Boat fuel emissions, equipment operations"
    },
    "fishing_pond": {
        "co2e_kg": 0.1,
        "low_carbon_score": 5.0,
        "notes": "Natural or semi-natural water body"
    },
    "fitness_center": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Equipment, HVAC, showers, lighting"
    },
    "golf_course": {
        "co2e_kg": 4.0,
        "low_carbon_score": 60.0,
        "notes": "Irrigation, maintenance equipment, clubhouse"
    },
    "gym": {
        "co2e_kg": 2.0,
        "low_carbon_score": 45.0,
        "notes": "Equipment, HVAC, showers, extended hours"
    },
    "ice_skating_rink": {
        "co2e_kg": 5.0,
        "low_carbon_score": 65.0,
        "notes": "Ice refrigeration, resurfacing, HVAC"
    },
    "playground": {
        "co2e_kg": 0.0,
        "low_carbon_score": 0.0,
        "notes": "Outdoor equipment, no energy use"
    },
    "ski_resort": {
        "co2e_kg": 10.0,
        "low_carbon_score": 80.0,
        "notes": "Lifts, snowmaking, lodge operations"
    },
    "sports_activity_location": {
        "co2e_kg": 1.5,
        "low_carbon_score": 38.0,
        "notes": "Variable based on specific sport"
    },
    "sports_club": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Multiple facilities, club operations"
    },
    "sports_coaching": {
        "co2e_kg": 0.5,
        "low_carbon_score": 20.0,
        "notes": "Minimal facilities, often outdoor"
    },
    "sports_complex": {
        "co2e_kg": 3.0,
        "low_carbon_score": 55.0,
        "notes": "Multiple venues, pools, courts, HVAC"
    },
    "stadium": {
        "co2e_kg": 6.5,
        "low_carbon_score": 70.0,
        "notes": "Large venue operations, field lighting, screens"
    },
    "swimming_pool": {
        "co2e_kg": 2.8,
        "low_carbon_score": 53.0,
        "notes": "Water heating, filtration, chemicals, HVAC"
    },
    # Transportation
    "international_airport": {
        "co2e_kg": 2.5,
        "low_carbon_score": 50.0,
        "notes": "Terminal HVAC, retail, restaurants, attractions, 24/7 operations"
    }
}