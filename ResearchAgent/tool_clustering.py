import math
from config import SINGAPORE_GEOCLUSTERS_BOUNDARIES

def calculate_geo_cluster(lat: float, lng: float) -> str:
    """
    Calculate Singapore geo cluster based on defined boundaries.
    If point doesn't fall within any boundary, find the closest cluster.
    """
    if not lat or not lng:
        return "unknown"
    
    # Singapore bounds check
    if not (1.1 <= lat <= 1.5 and 103.6 <= lng <= 104.1):
        return "unknown"
    
    # First, check if point falls within any defined boundary
    for cluster_name, bounds in SINGAPORE_GEOCLUSTERS_BOUNDARIES.items():
        if (bounds["lat_min"] <= lat <= bounds["lat_max"] and 
            bounds["lon_min"] <= lng <= bounds["lon_max"]):
            # Special handling: prefer downtown over overlapping regions
            if cluster_name == "central" and "downtown" in SINGAPORE_GEOCLUSTERS_BOUNDARIES:
                downtown = SINGAPORE_GEOCLUSTERS_BOUNDARIES["downtown"]
                if (downtown["lat_min"] <= lat <= downtown["lat_max"] and 
                    downtown["lon_min"] <= lng <= downtown["lon_max"]):
                    return "downtown"
            return cluster_name
    
    # If point doesn't fall within any boundary, find closest cluster
    return find_closest_cluster(lat, lng)

def find_closest_cluster(lat: float, lng: float) -> str:
    """
    Find the closest cluster based on distance to cluster centers.
    """
    min_distance = float('inf')
    closest_cluster = "unknown"
    
    for cluster_name, bounds in SINGAPORE_GEOCLUSTERS_BOUNDARIES.items():
        # Calculate center of the cluster
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lng = (bounds["lon_min"] + bounds["lon_max"]) / 2
        
        # Calculate Euclidean distance (simplified for small areas)
        # For more accuracy, use Haversine formula
        distance = math.sqrt((lat - center_lat)**2 + (lng - center_lng)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster_name
    
    return closest_cluster

# Test function
def test_clustering():
    """Test with known Singapore locations"""
    test_locations = [
        (1.4344838, 103.77948479999999, "Citrus By The Pool"),
        (1.4281829, 103.7674324, "Cheval Cafe Bar Bistro"),
        (1.4282523, 103.7993501, "elemen"),
        (1.4032692, 103.9132265, "Anna's"),
        (1.3969192, 103.9212159, "Daruma Tavern Punggol"),
        (1.3858487, 103.8975509, "Big Prawn Noodle"),
        (1.3104434999999999, 103.94609229999999, "Fico"),
        (1.3087539, 103.9027938, "La Bottega Enoteca"),
        (1.3148971, 103.9110412, "The Brewing Ground"),
        (1.3130167, 103.8565232, "MTR Singapore"),
        (1.2805121, 103.8503809, "Lau Pa Sat"),
        (1.2803361, 103.844767, "Maxwell Food Centre"),
        (1.3629771, 103.7645031, "iO Italian Osteria - HillV2 (Singapore)"),
        (1.3398183000000001, 103.73076979999999, "Eden | Chinese Garden | Halal-Certified"),
        (1.3383983, 103.75594319999999, "Laifaba Authentic Wood-Fired Roast & Noodles"),
        (1.2893135, 103.84829479999999, "JUMBO Seafood - The Riverwalk"),
    ]
    
    for lat, lng, name in test_locations:
        cluster = calculate_geo_cluster(lat, lng)
        print(f"{name}: {cluster}")