import collections
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    *result_sets: List[List[Dict[str, Any]]], 
    k: int = 60, 
    top_n: int = 25
) -> List[Dict[str, Any]]:
    """
    Combines multiple result sets using Reciprocal Rank Fusion (RRF).
    Each result set is a list of dicts with at least a 'url' key.
    
    Score(d) = sum_{r in rankings} 1 / (k + rank(d, r))
    """
    scores = collections.defaultdict(float)
    item_cache = {}

    for ranking in result_sets:
        for i, item in enumerate(ranking):
            url = item["url"]
            # Normalize URL for matching (trailing slash)
            url = url if url.endswith('/') else url + '/'
            
            # 1-indexed rank
            scores[url] += 1.0 / (k + (i + 1))
            
            # Cache the full item for later hydration (prefer the one with more metadata)
            if url not in item_cache or len(item.get("description", "")) > len(item_cache[url].get("description", "")):
                item_cache[url] = item

    # Sort by score descending
    sorted_urls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for url, score in sorted_urls[:top_n]:
        item = item_cache[url]
        item["rrf_score"] = score
        fused_results.append(item)
        
    return fused_results
