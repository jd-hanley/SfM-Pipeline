from dataclasses import dataclass
from collections import deque

from matching import ImageMatchPair

""" Track data structure contains information about each connected component in the feature graph """
@dataclass
class Track:
    track_id: int
    observations: dict[int, int]

"""
Given pairwise feature matches between images, build an adjacency list representation feature graph
Input: 
    all_matches: dict[tuple[int, int], ImagePairMatch], dictionary mapping image pairs to structs describing their matches
Output:
    graph: adjacency list representation where nodes are (image_id, keypoint_index) tuples
"""
def build_match_graph(all_matches: dict[tuple[int,int], ImageMatchPair]) -> dict[tuple[int,int], list[tuple[int,int]]]:

    graph = {}

    for (id1, id2), match in all_matches.items():

        for kp_idx1, kp_idx2 in zip(
            match.keypoint_indices1,
            match.keypoint_indices2
        ):
            
            node1 = (id1, int(kp_idx1))
            node2 = (id2, int(kp_idx2))

            if node1 in graph:
                graph[node1].append(node2)
            else:
                graph[node1] = [node2]
            
            if node2 in graph:
                graph[node2].append(node1)
            else:
                graph[node2] = [node1]
        
    return graph

"""
Run BFS from a given start node and return all nodes in the connected component
Input:
    graph: adjacency list representation where nodes are (image_id, keypoint_index) tuples
    visited: set[tuple[int,int]], set indicating whether a node has already been visited
    start: tuple[int,int], start node for the search
Output:
    component: list[tuple[int,int]], list of nodes in the connected component
"""
def find_connected_component(start, graph, visited: set[tuple[int,int]]) -> list[tuple[int,int]]:

    component = []

    # Enqueue the start node
    q = deque()
    q.append(start)
    visited.add(start)

    while q:

        curr = q.popleft()
        
        component.append(curr)

        for neighbor in graph[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    
    return component

""" Convert connected components to a Track if valid 
Input:
    component: list[tuple[int,int]], list of nodes in the connected component
    track_id: int
Output:
    track: Track
"""
def component_to_track(component: list[tuple[int,int]], track_id: int) -> Track | None:

    # Note that for a track to be valid it must have 2+ observations and maximum one keypoint per image

    observations = {}

    for image_id, kp_idx in component:
        
        # If we have matches within an image, invalidate this component
        if image_id in observations:
            return None
        
        observations[image_id] = kp_idx

    # If we didn't match between multiple images then this is useless
    if len(observations) < 2:
        return None

    temp = Track(track_id, observations)
    return temp

"""
Given pairwise matches between images, build the feature graph and build tracks
Input:
    all_matches: all_matches: dict[tuple[int, int], ImagePairMatch], dictionary mapping image pairs to structs describing their matches
Output:
    tracks: list[Track], list of valid track structs that provide location of a keypoint across all images that share that point
"""
def build_tracks(all_matches: dict[tuple[int, int], ImageMatchPair]) -> list[Track]:

    graph = build_match_graph(all_matches)

    visited = set()
    tracks = []
    count = 0

    for node in graph.keys():

        if node in visited:
            continue

        component = find_connected_component(node, graph, visited)
        track = component_to_track(component, count)
        
        if track is not None:
            tracks.append(track)
            count += 1
    
    return tracks

