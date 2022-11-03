# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

from typing import NamedTuple, Tuple
import os
import random
import time
from functools import partial
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import osmnx as ox
import folium
import seaborn as sns
import warnings

from dwave.system import LeapHybridDQMSampler
from cvrp.cvrp import CVRP

ox.settings.use_cache = True
ox.settings.overpass_rate_limit = False

depot_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Depot Location.png"))
force_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "Force Location.png"))
address = 'Cambridge St, Rockhampton QLD 4700, Australia'
depot_icon = folium.CustomIcon(depot_icon_path, icon_size=(32, 37))

class RoutingProblemParameters(NamedTuple):
    """Structure to hold all provided problem parameters.

    Args:
        folium_map: Folium map with locations already shown on map.
        map_network: `nx.MultiDiGraph` where nodes and edges represent locations and routes.
        depot_id: Node ID of the depot location.
        client_subset: client_subset: List of client IDs in the map's graph.
        num_clients: Number of locations to be visited.
        num_vehicles: Number of vehicles to deploy on routes.
        sampler_type: Sampler type to use in solving CVRP.
        time_limit: Time limit in seconds to run optimization for.

    """
    folium_map: folium.Map
    map_network: nx.MultiDiGraph
    depot_id: int
    client_subset: list
    num_clients: int
    num_vehicles: int
    sampler_type: str
    time_limit: float

def _cost_between_nodes(dijkstra_paths_and_lengths: dict, p1, p2, start_node: int, end_node: int) -> float:
    return dijkstra_paths_and_lengths[start_node][0][end_node]

def _cost_between_nodes_haversine(p1, p2, start, end) -> float:
    radius_earth = 6371000 # meters
    lat1_rad, lat2_rad = np.deg2rad((p1[0], p2[0]))
    diff_lat_rad, diff_lon_rad = np.deg2rad((p2[0] - p1[0], p2[1] - p1[1]))
    return 2 * radius_earth * np.arcsin(
        np.sqrt(np.sin(diff_lat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(diff_lon_rad/2)**2)
    )

def _map_network_from_address(address: str, network_type = "drive") -> nx.MultiDiGraph:
    """Generate a sparse `nx.MultiDiGraph` network from address.

    Args:
        address: String address to generate network from.

    Returns:
        `nx.MultiDiGraph` object of map network, retaining the largest
            strongly connected component of the graph.

    """
    G = ox.graph_from_address(address, dist=1000, network_type=network_type)
    G = ox.utils_graph.get_largest_component(G, strongly=True)
    return G

def _all_pairs_dijkstra_dict(G: nx.MultiDiGraph) -> dict:
    return dict(nx.all_pairs_dijkstra(G, weight='length'))

def _build_node_index_map(G: nx.MultiDiGraph) -> dict:
    return dict(enumerate(G.nodes(data=True)))

def _find_node_index_central_to_network(node_index_map: dict) -> int:
    coordinates = np.zeros((len(node_index_map), 2))
    for node_index, node in node_index_map.items():
        coordinates[node_index][0] = node[1]['y']
        coordinates[node_index][1] = node[1]['x']
    
    centroid = np.sum(coordinates, 0) / len(node_index_map)
    kd_tree = cKDTree(coordinates)
    return kd_tree.query(centroid)[1]

def _select_client_nodes(G: nx.MultiDiGraph, depot_id: int, num_clients: int) -> list:
    """Select a subset of nodes to represent client and depot locations.

    Args:
        G: Map network to draw subset from.
        depot_id: Node ID of the depot location.
        num_clients: Number of client locations desired in CVRP problem.

    Returns:
        List containing subset of nodes of length `num_clients`.

    """
    random.seed(num_clients)
    graph_copy = G.copy()
    graph_copy.remove_node(depot_id)
    return random.sample(list(graph_copy.nodes), num_clients)

def generate_mapping_information(num_clients: int) -> Tuple[nx.MultiDiGraph, int, list]:
    """Return `nx.MultiDiGraph` with client demand, depot id in graph, client ids in graph.

    Args:
        num_clients: Number of locations to be visited in total.

    Returns:
        map_network: `nx.MultiDiGraph` where nodes and edges represent locations and routes.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.

    """
    map_network = _map_network_from_address(address=address)

    node_index_map = _build_node_index_map(map_network)

    depot_id = node_index_map[_find_node_index_central_to_network(node_index_map)][0]
    client_subset = _select_client_nodes(map_network, depot_id, num_clients = num_clients)

    for node_id in client_subset:
        map_network.nodes[node_id]['demand_water'] = random.choice([1,2])
        map_network.nodes[node_id]['demand_food'] = random.choice([1,2])
        map_network.nodes[node_id]['demand_other'] = random.choice([1,2])
    
        map_network.nodes[node_id]['demand'] = map_network.nodes[node_id]['demand_water'] +\
                                               map_network.nodes[node_id]['demand_food'] +\
                                               map_network.nodes[node_id]['demand_other']

    return map_network, depot_id, client_subset

def _map_network_as_cvrp(G: nx.MultiDiGraph, depot_id: int, client_subset: list, 
                         partial_cost_func: callable, num_vehicles: int) -> CVRP:
    """Generate CVRP instance from map network and select information.

    Args:
        G: Map network to build CVRP problem from.
        depot_id: Node ID of the depot location.
        client_subset: List of client IDs in the map's graph.
        partial_cost_func: Partial cost function to pass to CVRP.
        num_vehicles: Number of vehicles to deploy on routes.

    Returns:
        Instance of CVRP class.

    """
    demand = nx.get_node_attributes(G, 'demand')
    depot = {depot_id: (G.nodes[depot_id]['y'], G.nodes[depot_id]['x'])}
    clients = {client_id: (G.nodes[client_id]['y'], G.nodes[client_id]['x']) for client_id in client_subset}

    cvrp = CVRP(cost_function=partial_cost_func)
    cvrp.add_depots(depot)
    cvrp.add_forces(clients, demand)
    cvrp.add_vehicles({k: -(-sum(demand.values()) // num_vehicles) for k in range(num_vehicles)})

    return cvrp


def show_locations_on_initial_map(G: nx.MultiDiGraph, depot_id: int, client_subset: list) -> folium.folium.Map:
    """Prepare map to be rendered initially on app screen.

    Args:
        G: `nx.MultiDiGraph` to build map from.
        depot_id: Node ID of the depot location.
        client_subset: client_subset: List of client IDs in the map's graph.

    Returns:

    """
    folium_map = ox.plot_graph_folium(G, opacity=0.0)

    folium.Marker(
        (G.nodes[depot_id]['y'], G.nodes[depot_id]['x']),
        tooltip=folium.map.Tooltip(text="Depot", style="font-size: 1.4rem;"), icon=depot_icon
    ).add_to(folium_map)

    for force_id in client_subset:
        if force_id != depot_id:
            location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
            folium.Marker(
                (G.nodes[force_id]['y'], G.nodes[force_id]['x']),
                tooltip=folium.map.Tooltip(text=f"large: {G.nodes[force_id]['demand_water'] * 100} \
                          <br> small: {G.nodes[force_id]['demand_food'] * 100} <br> \
                          fragiles: {G.nodes[force_id]['demand_other'] * 100}", style="font-size: 1.4rem;"),
                icon=location_icon
            ).add_to(folium_map)

    folium.plugins.Fullscreen().add_to(folium_map)
    return folium_map


def _plot_solution_routes_on_drone_map(folium_map, G, solution: dict, depot_id: int) -> folium.folium.Map:
    """Generate interactive folium map for drone routes given solution dictionary.

    Args:
        G: Map network to plot.
        solution: Solution returned by CVRP.
        depot_id: Node ID of the depot location.

    Returns:
        `folium.folium.Map` object,  dictionary with solution cost information.

    """
    solution_cost_information = {}
    palette = sns.color_palette("colorblind", len(solution)).as_hex()

    locations = {}
    for vehicle_id, route_network in solution.items():
        solution_cost_information[vehicle_id + 1] = {
            "optimized_cost": 0,
            "forces_serviced": len(route_network.nodes) - 1,
            "water": 0,
            "food": 0,
            "other": 0,
        }

        for node in route_network.nodes:
            locations.update({node:(G.nodes[node]['y'], G.nodes[node]['x'])})
            if node != depot_id:
                location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
                folium.Marker(
                    locations[node],
                    tooltip=folium.map.Tooltip(text=f"large: {G.nodes[node]['demand_water'] * 100} \
                                                    <br> small: {G.nodes[node]['demand_food'] * 100} <br> \
                                                    fragiles: {G.nodes[node]['demand_other'] * 100} <br> vehicle: {vehicle_id+1}",
                                               style="font-size: 1.4rem;"),
                    icon=location_icon
                ).add_to(folium_map)

                solution_cost_information[vehicle_id + 1]["water"] += G.nodes[node]['demand_water'] * 100
                solution_cost_information[vehicle_id + 1]["food"] += G.nodes[node]['demand_food'] * 100
                solution_cost_information[vehicle_id + 1]["other"] += G.nodes[node]['demand_other'] * 100

        route_color = palette.pop()
        for edge in route_network.edges:
            solution_cost_information[vehicle_id + 1]["optimized_cost"] += _cost_between_nodes_haversine(
                locations[edge[0]], locations[edge[1]], edge[0], edge[1]
            )
            folium.PolyLine(
                [locations[node] for node in edge], color=route_color
            ).add_to(folium_map)

    return folium_map, solution_cost_information


def _plot_solution_routes_on_street_map(folium_map, G, solution: dict, depot_id: int, dijkstra_paths: dict) -> folium.folium.Map:
    """Generate interactive folium map for truck routes given solution dictionary.

    Args:
        G: Map network to plot.
        solution: Solution returned by CVRP.
        depot_id: Node ID of the depot location.
        dijkstra_paths: Dictionary containing both paths and path lengths between any two nodes.

    Returns:
        `folium.folium.Map` object, dictionary with solution cost information.

    """
    solution_cost_information = {}
    palette = sns.color_palette("colorblind", len(solution)).as_hex()

    for vehicle_id, route_network in solution.items():
        solution_cost_information[vehicle_id + 1] = {
            "optimized_cost": 0,
            "forces_serviced": len(route_network.nodes) - 1,
            "water": 0,
            "food": 0,
            "other": 0,
        }

        for node in route_network.nodes:
            if node != depot_id:
                location_icon = folium.CustomIcon(force_icon_path, icon_size=(32, 37))
                folium.Marker(
                    (G.nodes[node]['y'], G.nodes[node]['x']),
                    tooltip=folium.map.Tooltip(text=f"water: {G.nodes[node]['demand_water'] * 100} \
                                                    <br> food: {G.nodes[node]['demand_food'] * 100} <br> \
                                                    other: {G.nodes[node]['demand_other'] * 100} <br> vehicle: {vehicle_id+1}",
                                               style="font-size: 1.4rem;"),
                    icon=location_icon
                ).add_to(folium_map)

                solution_cost_information[vehicle_id + 1]["water"] += G.nodes[node]['demand_water'] * 100
                solution_cost_information[vehicle_id + 1]["food"] += G.nodes[node]['demand_food'] * 100
                solution_cost_information[vehicle_id + 1]["other"] += G.nodes[node]['demand_other'] * 100

        route_color=palette.pop()
        routes = [dijkstra_paths[start][1][end] for start,end in route_network.edges]

        solution_cost_information[vehicle_id + 1]["optimized_cost"] += sum([dijkstra_paths[start][0][end] for start,end in route_network.edges])

        for route in routes:
            folium_map = ox.plot_route_folium(G, route=route, route_map=folium_map, fit_bounds=False, color=route_color, popup_attribute='length')

    return folium_map, solution_cost_information


def generate_solution_map_drone_network(problem_parameters: RoutingProblemParameters) -> dict:
    """Generate map with solution routes plotted, map centered on depot location, for drone routes.

    Args:
        problem_parameters: NamedTuple that specifies all problem details.

    Returns:
        dict containing solved state map, solution information.

    """
    start_time = time.perf_counter()
    cvrp = _map_network_as_cvrp(
        problem_parameters.map_network, 
        problem_parameters.depot_id, 
        problem_parameters.client_subset, 
        _cost_between_nodes_haversine, 
        problem_parameters.num_vehicles
    )
    solved_state_cvrp = _solved_state_cvrp(
        cvrp,
        problem_parameters.sampler_type, 
        disentangle=False, 
        time_limit=problem_parameters.time_limit
    )

    wall_clock_time = time.perf_counter() - start_time

    solution_map, solution_cost_information = _plot_solution_routes_on_drone_map(
        problem_parameters.folium_map, 
        problem_parameters.map_network, 
        solved_state_cvrp.solution, 
        problem_parameters.depot_id
    )

    return {
        "map": solution_map,
        "wall_clock_time": wall_clock_time,
        "solution_cost": solution_cost_information
    }

def generate_solution_map_street_network(problem_parameters: RoutingProblemParameters) -> dict:
    """Generate map with solution routes plotted, map centered on depot location, for truck routes.

    Args:
        problem_parameters: NamedTuple that specifies all problem details.

    Returns:
        dict containing solved state map, solution information.

    """
    start_time = time.perf_counter()
    paths_and_lengths = _all_pairs_dijkstra_dict(problem_parameters.map_network)

    partial_cost_func = partial(_cost_between_nodes, paths_and_lengths)

    cvrp = _map_network_as_cvrp(
        problem_parameters.map_network, 
        problem_parameters.depot_id, 
        problem_parameters.client_subset, 
        partial_cost_func, 
        problem_parameters.num_vehicles
    )

    solved_state_cvrp = _solved_state_cvrp(
        cvrp, 
        problem_parameters.sampler_type, 
        disentangle=False, 
        time_limit=problem_parameters.time_limit
    )
    wall_clock_time = time.perf_counter() - start_time

    solution_map, solution_cost_information = _plot_solution_routes_on_street_map(
        problem_parameters.folium_map, 
        problem_parameters.map_network, 
        solved_state_cvrp.solution, 
        problem_parameters.depot_id, 
        paths_and_lengths
    )

    return {
        "map": solution_map,
        "wall_clock_time": wall_clock_time,
        "solution_cost": solution_cost_information,
    }

def _solved_state_cvrp(cvrp, sampler_type, disentangle=True, time_limit=None):
    if sampler_type == 'Classical (K-Means)':
        cvrp.cluster(sampler='kmeans', step_size=0.6, time_limit=time_limit)
    if sampler_type == "Quantum Hybrid (DQM)":
        try:
            cvrp.cluster(sampler=LeapHybridDQMSampler(), lagrange={'capacity': 1.0}, time_limit=time_limit)
        except ValueError:
            warnings.warn("Defaulting to minimum time limit for Leap Hybrid DQM Sampler.")
            cvrp.cluster(
                sampler=LeapHybridDQMSampler(), lagrange={'capacity': 1.0}, time_limit=None
            )
    cvrp.solve_tsp_heuristic(disentangle=disentangle)
    return cvrp
