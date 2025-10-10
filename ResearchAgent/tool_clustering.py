def calculate_geo_cluster(lat: float, lng: float) -> str:
    """
    Calculate Singapore geo cluster based purely on geographic regions.
    
    Singapore regions based on actual geographic boundaries:
    - Central: Downtown Core, Marina Bay, Orchard
    - North: Woodlands, Sembawang, Yishun  
    - South: Sentosa, HarbourFront, Telok Blangah
    - East: Changi, Pasir Ris, Tampines
    - West: Jurong, Tuas, Pioneer
    - North-East: Punggol, Sengkang, Hougang
    - North-West: Bukit Panjang, Choa Chu Kang, Kranji
    - South-East: Marine Parade, East Coast, Katong
    - South-West: Queenstown, Buona Vista, Clementi
    """
    if not lat or not lng:
        return "unknown"
    
    # Singapore bounds check
    if not (1.1 <= lat <= 1.5 and 103.6 <= lng <= 104.1):
        return "unknown"
    
    # Define boundary thresholds based on Singapore's actual geography
    LAT_SOUTH = 1.27
    LAT_CENTRAL_SOUTH = 1.27  
    LAT_CENTRAL_NORTH = 1.32
    LAT_NORTH = 1.37
    
    LNG_WEST = 103.75
    LNG_CENTRAL_WEST = 103.82
    LNG_CENTRAL_EAST = 103.87
    LNG_EAST = 103.95
    
    # Central region first (Downtown, Marina, Orchard area)
    if (LAT_CENTRAL_SOUTH <= lat <= LAT_CENTRAL_NORTH and 
        LNG_CENTRAL_WEST <= lng <= LNG_CENTRAL_EAST):
        return "central"
    
    # Determine region based on coordinates
    if lat > LAT_NORTH:
        if lng < LNG_WEST:
            return "north-west"
        elif lng > LNG_EAST:
            return "north-east"
        else:
            return "north"
    elif lat < LAT_SOUTH:
        if lng < LNG_WEST:
            return "south-west"
        elif lng > LNG_EAST:
            return "south-east"
        else:
            return "south"
    else:  # Middle latitude
        if lng < LNG_WEST:
            return "west"
        elif lng > LNG_EAST:
            return "east"
        else:
            # Areas between central and outer
            if lng < LNG_CENTRAL_WEST:
                return "west"
            elif lng > LNG_CENTRAL_EAST:
                return "east"
            else:
                return "central"


def group_places_by_proximity(places: list, min_cluster_size: int = 3) -> dict:
    """
    Group places into geographic clusters based on their locations.
    
    Rules:
    1. Group places in same geographic region
    2. Merge small clusters with nearest larger cluster
    3. Ensure logical geographic groupings
    
    Args:
        places: List of places with geo coordinates
        min_cluster_size: Minimum places to maintain as separate cluster
    
    Returns:
        Dictionary with clustering results and reasoning
    """
    # Step 1: Assign each place to its geographic cluster
    clusters = {}
    unassigned = []
    
    for place in places:
        if place.get('geo') and place['geo'].get('latitude'):
            cluster_id = calculate_geo_cluster(
                place['geo']['latitude'],
                place['geo']['longitude']
            )
            if cluster_id != "unknown":
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(place)
            else:
                unassigned.append(place)
        else:
            unassigned.append(place)
    
    # Step 2: Define geographic adjacency for merging
    adjacency = {
        "central": ["north", "south", "east", "west"],  # Central connects to cardinal directions
        "north": ["north-east", "north-west", "central"],
        "south": ["south-east", "south-west", "central"],
        "east": ["north-east", "south-east", "central"],
        "west": ["north-west", "south-west", "central"],
        "north-east": ["north", "east"],
        "north-west": ["north", "west"],
        "south-east": ["south", "east"],
        "south-west": ["south", "west"]
    }
    
    # Step 3: Merge small clusters with adjacent regions
    final_clusters = {}
    merge_log = []
    
    # Sort by size to prioritize keeping larger clusters
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, cluster_places in sorted_clusters:
        if len(cluster_places) >= min_cluster_size:
            # Keep as separate cluster
            final_clusters[cluster_id] = cluster_places
        else:
            # Find best adjacent cluster to merge with
            best_merge = None
            best_size = 0
            
            for adjacent_id in adjacency.get(cluster_id, []):
                if adjacent_id in final_clusters:
                    if len(final_clusters[adjacent_id]) > best_size:
                        best_merge = adjacent_id
                        best_size = len(final_clusters[adjacent_id])
            
            if best_merge:
                final_clusters[best_merge].extend(cluster_places)
                merge_log.append(f"Merged {cluster_id} ({len(cluster_places)} places) into {best_merge}")
            else:
                # No suitable merge, keep as small cluster
                final_clusters[cluster_id] = cluster_places
    
    # Step 4: Handle unassigned places
    if unassigned and final_clusters:
        # Add to central cluster if it exists, otherwise largest cluster
        if "central" in final_clusters:
            final_clusters["central"].extend(unassigned)
        else:
            largest_cluster = max(final_clusters.keys(), key=lambda k: len(final_clusters[k]))
            final_clusters[largest_cluster].extend(unassigned)
    
    # Step 5: Generate clustering reasoning
    cluster_stats = []
    for cluster_id, cluster_places in final_clusters.items():
        cluster_stats.append({
            "cluster_id": cluster_id,
            "region_name": _get_region_name(cluster_id),
            "place_count": len(cluster_places),
            "places": cluster_places,
            "characteristics": _get_cluster_characteristics(cluster_id, cluster_places)
        })
    
    # Sort by place count for presentation
    cluster_stats.sort(key=lambda x: x['place_count'], reverse=True)
    
    return {
        "total_places": len(places),
        "clusters_formed": len(cluster_stats),
        "clustering_log": merge_log,
        "clusters": cluster_stats,
        "summary": _generate_clustering_summary(cluster_stats)
    }


def _get_region_name(cluster_id: str) -> str:
    """Get human-readable region name."""
    region_names = {
        "central": "Central Singapore (City/Marina/Orchard)",
        "north": "Northern Singapore (Woodlands/Yishun)",
        "south": "Southern Singapore (Sentosa/HarbourFront)",
        "east": "Eastern Singapore (Changi/Tampines)",
        "west": "Western Singapore (Jurong)",
        "north-east": "North-Eastern Singapore (Punggol/Sengkang)",
        "north-west": "North-Western Singapore (Bukit Panjang/Kranji)",
        "south-east": "South-Eastern Singapore (East Coast/Katong)",
        "south-west": "South-Western Singapore (Clementi/Queenstown)"
    }
    return region_names.get(cluster_id, f"Region: {cluster_id}")


def _get_cluster_characteristics(cluster_id: str, places: list) -> dict:
    """Analyze characteristics of places in cluster."""
    # Count place types
    type_counts = {}
    for place in places:
        place_type = place.get('type', 'unknown')
        type_counts[place_type] = type_counts.get(place_type, 0) + 1
    
    # Calculate geographic spread within cluster
    if places:
        lats = [p['geo']['latitude'] for p in places if p.get('geo')]
        lngs = [p['geo']['longitude'] for p in places if p.get('geo')]
        
        if lats and lngs:
            lat_spread = max(lats) - min(lats)
            lng_spread = max(lngs) - min(lngs)
            spread_km = ((lat_spread * 111) ** 2 + (lng_spread * 111) ** 2) ** 0.5
        else:
            spread_km = 0
    else:
        spread_km = 0
    
    return {
        "dominant_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
        "type_distribution": type_counts,
        "geographic_spread_km": round(spread_km, 2),
        "density": "dense" if spread_km < 2 else "moderate" if spread_km < 5 else "spread out"
    }


def _generate_clustering_summary(cluster_stats: list) -> str:
    """Generate summary of clustering results."""
    if not cluster_stats:
        return "No clusters formed"
    
    summaries = []
    
    # Identify main clusters
    main_clusters = [c for c in cluster_stats if c['place_count'] >= 5]
    small_clusters = [c for c in cluster_stats if c['place_count'] < 5]
    
    if main_clusters:
        cluster_names = [c['cluster_id'] for c in main_clusters]
        summaries.append(f"Main attraction areas: {', '.join(cluster_names)}")
    
    if small_clusters:
        summaries.append(f"{len(small_clusters)} minor area(s) with fewer attractions")
    
    # Note geographic spread
    regions_covered = set(c['cluster_id'] for c in cluster_stats)
    if "central" in regions_covered:
        summaries.append("Strong concentration in central Singapore")
    
    cardinal_regions = {"north", "south", "east", "west"} & regions_covered
    if len(cardinal_regions) >= 3:
        summaries.append("Attractions spread across multiple regions of Singapore")
    elif len(cardinal_regions) <= 1:
        summaries.append("Attractions concentrated in limited geographic area")
    
    return ". ".join(summaries)