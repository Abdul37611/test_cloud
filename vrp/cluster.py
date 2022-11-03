from itertools import combinations
from statistics import variance
from typing import Dict

import pandas as pd
import numpy as np
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler

from .tsp import TSP, LoadBalancer


def two_stage_cluster(
    data: pd.DataFrame,
    dur_mat: pd.DataFrame,
    meta: Dict,
    balance=''
) -> pd.DataFrame:

    data = data.reset_index(drop=True).set_index('order_id')

    # First stage cluster based on pre-routing
    data = first_stage_cluster(data, dur_mat, meta, balance)

    # Second stage cluster
    orders_1 = data[data['cluster'].notnull()]
    orders_2 = data[data['cluster'].isna()]
    cqm = construct_second_stage_cqm(data, orders_1, orders_2, dur_mat, meta)
    sample = LeapHybridCQMSampler().sample_cqm(cqm).filter(
        lambda row: row.is_feasible)
    assignments = dict(
        entry[0] for entry in sample.first.sample.items() if entry[1] == 1)
    data['cluster'] = data.index.map(assignments)

    return data


def first_stage_cluster(
    data: pd.DataFrame,
    dur_mat: pd.DataFrame,
    meta: Dict,
    balance=''
) -> pd.DataFrame:

    remaining = data[data.apply(
        lambda row: row['distance'] > meta['distance_threshold'] * 1000
        and row['dispense_type'] == 'FLEET_DELIVERY', axis=1)]

    if len(remaining) == 0:
        data['cluster'] = np.nan
        return data

    subsets = {}
    for name, van in meta['vans'].items():
        subsets[name] = remaining[remaining.apply(
            lambda row: row['tw_start'] in van.slots_served, axis=1)]
    vans = sorted(meta['vans'], key=lambda x: len(subsets[x]))

    routes = {}
    solver = TSP(remaining, dur_mat, meta)
    while vans:

        van = vans.pop(0)
        subset = subsets[van]

        departure_time = meta['vans'][van].shift_start * 60
        shift_end = meta['vans'][van].shift_end * 60
        result = solver.solve_custom(subset, departure_time, shift_end)

        routes[van] = result
        orders = [order for order, _ in result[1:]]
        remaining = remaining.drop(index=orders)
        
        subsets = {}
        for name in vans:
            van = meta['vans'][name]
            subsets[name] = remaining[remaining.apply(
                lambda row: row['tw_start'] in van.slots_served, axis=1)]
        vans.sort(key=lambda x: len(subsets[x]))
    
    if len(remaining) > 0:
        print('Pre-routing was unable to assign routes to all required orders')
    
    if balance == 'naive':
        LoadBalancer(data, dur_mat, meta).naive_balance(routes)
    elif balance == 'centroid':
        LoadBalancer(data, dur_mat, meta).distance_based_balance(routes)
    elif balance == 'time':
        LoadBalancer(data, dur_mat, meta).minimize_average_travel_time(routes)

    for van, route in routes.items():
        orders = [order for order, _ in route[1:]]
        data.loc[orders, 'cluster'] = van

    return data


def construct_second_stage_cqm(
    data: pd.DataFrame,
    orders_1: pd.DataFrame,
    orders_2: pd.DataFrame,
    dur_mat: pd.DataFrame,
    meta: Dict
) -> ConstrainedQuadraticModel:

    cqm = ConstrainedQuadraticModel()
    for _, order in data.iterrows():
        cqm.add_discrete([(order.name, van) for van in meta['vans']])

    # Add distance metric
    objective = BinaryQuadraticModel('BINARY')
    for u, v in combinations(data.index, r=2):
        add_u, add_v = data.loc[u]['address_id'], data.loc[v]['address_id']
        if data.loc[u]['tw_start'] > data.loc[v]['tw_start']:
            cost = dur_mat.loc[add_v][add_u]
        elif data.loc[u]['tw_start'] < data.loc[v]['tw_start']:
            cost = dur_mat.loc[add_u][add_v]
        else:
            cost = (dur_mat.loc[add_u][add_v] + dur_mat.loc[add_v][add_u]) / 2
        for van in meta['vans']:
            objective.set_quadratic((u, van), (v, van), cost)
    cqm.set_objective(objective)

    # Add time window constraints
    add_time_window_constraints(cqm, orders_2, meta)

    # Points already assigned a cluster
    pre = []
    for _, order in orders_1.iterrows():
        pre.append(((order.name, order['cluster']), 1))
    cqm.add_constraint(pre, '==', len(pre))

    return cqm


def add_time_window_constraints(
    cqm: ConstrainedQuadraticModel,
    data: pd.DataFrame,
    meta: Dict
):
    for van in meta['vans']:
        slots = meta['vans'][van].slots_served
        unservable_orders = []
        for _, order in data.iterrows():
            if order['tw_start'] not in slots:
                unservable_orders.append(((order.name, van), 1))
        cqm.add_constraint(unservable_orders, '==', 0)
