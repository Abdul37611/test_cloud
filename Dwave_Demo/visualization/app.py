# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 

from collections import defaultdict
from jinja2 import Template
import streamlit as st
from streamlit_folium import folium_static
from map_network import (generate_solution_map_street_network,
                         generate_solution_map_drone_network,
                         generate_mapping_information,
                         show_locations_on_initial_map,
                         RoutingProblemParameters)

map_width, map_height = 1000, 600

def render_solution_stats(problem_size = None, 
                          search_space = None,
                          wall_clock_time = None,
                          forces = None,
                          vehicles= None) -> str:
    with open("app_customization/solution_stats.html") as stats:
        template = Template(stats.read())
        return template.render(
            problem_size = problem_size,
            search_space = search_space,
            wall_clock_time = wall_clock_time,
            num_forces = forces,
            num_vehicles = vehicles
        )

def render_solution_cost(solution_cost_information: dict, total_cost_information: dict):
    with open("app_customization/solution_cost.html") as cost:
        template = Template(cost.read())
        return template.render(
            solution_cost_information=solution_cost_information,
            total_cost_information=total_cost_information
        )

st.set_page_config(layout="wide")



with open("app_customization/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


with open("app_customization/stylesheet.html") as css:
  stylesheet = css.read()

st.write(stylesheet, unsafe_allow_html=True)

with open("app_customization/header.html") as header:
  template = Template(header.read())
  header_html = template.render()

st.write(header_html, unsafe_allow_html=True)

vehicle_type = st.sidebar.radio(
  "Vehicle Type", ["Delivery Drones", "Trucks"]
)

sampler = st.sidebar.radio(
  "Optimization Method", ["Quantum Hybrid", "Classical"]
)

if sampler == "Quantum Hybrid":
    sampler_type="Quantum Hybrid (DQM)"
elif sampler == "Classical":
    sampler_type="Classical (K-Means)"

num_clients = st.sidebar.slider(
    "Number of delivery locations", 10, 100, 10, 1
)

num_vehicles = st.sidebar.slider(
    "Number of vehicles to deploy", 1, 10, 2, 1
)

time_limit = st.sidebar.number_input("Optimization time limit (s)", min_value=5.0, value=5.0)

map_network, depot_id, force_locations = generate_mapping_information(num_clients)
initial_map = show_locations_on_initial_map(map_network, depot_id, force_locations)

routing_problem_parameters = RoutingProblemParameters(
    folium_map=initial_map, map_network=map_network, depot_id=depot_id,
    client_subset=force_locations, num_clients=num_clients,
    num_vehicles=num_vehicles, sampler_type=sampler_type, time_limit=time_limit
)

run_button = st.sidebar.button("Run Optimization", key="run")

if not run_button:
    folium_static(initial_map, width=map_width, height=map_height)

def render_app(solution_and_map_function: callable):
        response = solution_and_map_function(routing_problem_parameters)
        solution_stats = render_solution_stats(
            problem_size = num_vehicles * num_clients,
            search_space = "{:.2e}".format(num_vehicles ** num_clients),
            wall_clock_time = "{:.3f}".format(response["wall_clock_time"]),
            forces = num_clients,
            vehicles = num_vehicles
        )
        folium_static(
            response["map"],
            width=map_width,
            height=map_height
        )
        st.write(solution_stats, unsafe_allow_html=True)

        solution_cost_information  = dict(sorted(response["solution_cost"].items()))
        total_cost = defaultdict(int)
        for _, cost_info_dict in solution_cost_information.items():
            for key, value in cost_info_dict.items():
                total_cost[key] += value

        solution_cost = render_solution_cost(
            solution_cost_information=solution_cost_information,
            total_cost_information=total_cost
        )
        st.write(solution_cost, unsafe_allow_html=True)

if vehicle_type == 'Delivery Drones':
    if run_button:
        render_app(generate_solution_map_drone_network)
else:
    if run_button:
        render_app(generate_solution_map_street_network)
