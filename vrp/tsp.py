from statistics import variance
from typing import Dict, List, NamedTuple, Set, Tuple

import pandas as pd

from .preprocess import Vehicle

SERVICE_TIME = 12


class Label(NamedTuple):
    S: Tuple[int]  # a list of visited vertices as a tuple so that it's hashable
    t: int  # the time elapsed
    P: float  # number of customers served or score collected
    i: int  # the order ID
    predecessor: 'Label'  # the predecessor


class TSP:
    """The TSP solver"""

    def __init__(self, data: pd.DataFrame, dur_mat: pd.DataFrame, meta: Dict, prize=False):
        self.data = data
        self.dur_mat = dur_mat
        self.meta = meta
        self.prize = prize

    def solve(self, van: str) -> List[Tuple[int, int]]:
        shift_start = self.meta['vans'][van].shift_start * 60
        shift_end = self.meta['vans'][van].shift_end * 60
        data = self.data.loc[self.data['cluster'] == van]
        return self.solve_custom(data, shift_start, shift_end)

    def solve_custom(
        self,
        data: pd.DataFrame,
        departure_time: int,
        shift_end: int
    ) -> List[Tuple[int, int]]:
        """The algorithm"""

        if len(data) == 0:
            return []

        max_P = 0
        max_l = None

        labels = dict((idx, {
            'extended': set(),
            'unextended': set()
        }) for idx in data.index)
        labels[0] = {
            'extended': set(),
            'unextended': {Label((), departure_time, 0, 0, None)}
        }  # for the depot

        E = {0}  # set of vertices to examine
        while E:
            i = E.pop()
            while labels[i]['unextended']:
                li = labels[i]['unextended'].pop()
                successors: Set[int] = set(
                    v for v in data.index if v != li.i and v not in li.S)
                for j in successors:
                    lj = self.extend(li, j, shift_end)
                    if lj:
                        self.insert(labels[j]['unextended'], lj)
                        if lj.P > max_P:  # keep track of the most customers visited
                            max_P = lj.P
                            max_l = lj
                    if labels[j]['unextended']:
                        E.add(j)
                labels[i]['extended'].add(li)

        # generate route backtracking from the best state
        route = [(max_l.i, max_l.t)]
        curr = max_l
        while (curr := curr.predecessor) is not None:
            route.insert(0, (curr.i, curr.t))
        return route

    def extend(self, label: Label, vertex: int, shift_end: int) -> Label:
        """Extends a state to a vertex"""

        # update visited vector
        S = label.S + (vertex,)

        # update time elapsed
        add_i, add_j = self.get_address(label.i), self.get_address(vertex)
        t_ij = self.dur_mat.loc[add_i][add_j]
        tw_start = self.data.loc[vertex]['tw_start'] * 60
        t = max(label.t + SERVICE_TIME + t_ij, tw_start)  # wait if arrives early

        # number of customers served
        if self.data.loc[vertex]['distance'] > self.meta['distance_threshold'] * 1000:
            P = (label.P + 100 / (t_ij + 1)) if self.prize else label.P + 100
        else:
            P = label.P + 1

        # verify feasibility
        tw_end = self.data.loc[vertex]['tw_end'] * 60
        if t > tw_end:  # make sure to arrive before the deadline
            return None

        # make sure to get back to the depot by shift end
        add_i, add_j = self.get_address(vertex), self.get_address(0)
        t_ij = self.dur_mat.loc[add_i][add_j]
        if t + SERVICE_TIME + t_ij > shift_end:
            return None

        # construct label
        return Label(S, t, P, vertex, label)

    def insert(self, labels: Set[Label], lj: Label):
        """Insert a label into a set of labels applying dominance rules"""

        for label in labels:
            # an existing state dominates the candidate state if the existing state
            # takes less time, while serving more customers
            if label.t <= lj.t and label.P >= lj.P:
                return

        labels.add(lj)

    def get_address(self, index: int) -> str:
        """Gets address based on address ID"""

        if index == 0:
            return self.meta['store_address_id']
        return self.data.loc[index]['address_id']


def print_result(result: List):

    print('     order_id   arrival_time\n----------------------------')
    for order in result[1:]:
        hour = int(order[1] / 60)
        minute = order[1] - hour * 60
        print(f'{order[0]}       {hour:02}:{minute:02}')
    print(f'\nNumber of orders served: {len(result)-1}')


class LoadBalancer:
    """Balances the load of pre-assigned routes"""

    def __init__(self, data: pd.DataFrame, dur_mat: pd.DataFrame, meta: Dict):
        self.data = data
        self.dur_mat = dur_mat
        self.meta = meta

    def try_insert(
        self,
        van: Vehicle,
        route: List[Tuple],
        order: pd.Series,
        index: int
    ) -> Tuple[bool, List[Tuple]]:

        pre = route[index]
        new_route = route.copy()

        add_i, add_j = self.get_address(pre[0]), self.get_address(order.name)
        t_ij = self.dur_mat.loc[add_i][add_j]
        tw_start = order['tw_start'] * 60
        t = max(pre[1] + SERVICE_TIME + t_ij, tw_start)  # wait if arrives early

        # verify feasibility
        tw_end = order['tw_end'] * 60
        if t > tw_end:  # make sure to arrive before the deadline
            return False, route

        new_stop = (order.name, t)
        new_route.insert(index + 1, new_stop)

        # cascade arrival times down the line
        for idx in range(index + 2, len(new_route)):

            idx_i, idx_j = new_route[idx - 1][0], new_route[idx][0]
            add_i, add_j = self.get_address(idx_i), self.get_address(idx_j)
            t_ij = self.dur_mat.loc[add_i][add_j]
            tw_start = self.data.loc[idx_j]['tw_start'] * 60
            t = max(new_route[idx - 1][1] + SERVICE_TIME + t_ij, tw_start)

            tw_end = self.data.loc[idx_j]['tw_end'] * 60
            if t > tw_end:  # make sure to arrive before the deadline
                return False, route

            new_route[idx] = (idx_j, t)

        # make sure to still return to the depot by shift end
        add_i, add_j = self.get_address(new_route[-1][0]), self.get_address(0)
        t_ij = self.dur_mat.loc[add_i][add_j]
        if t + SERVICE_TIME + t_ij > van.shift_end * 60:
            return False, route

        # return final result
        return True, new_route


    def insert(
        self,
        van: Vehicle,
        route: List[Tuple],
        order: pd.Series,
        travel_time_based=False
    ) -> Tuple[int, List[Tuple]]:

        tw_start, tw_end = order['tw_start'] * 60, order['tw_end'] * 60
        candidates = [
            idx for idx in range(len(route) - 1)
            if route[idx][1] + SERVICE_TIME < tw_end
            and route[idx + 1][1] > tw_start + SERVICE_TIME
        ]  # filter for possible insertion locations

        # check last location
        if (route[-1][1] + SERVICE_TIME < tw_end and
                van.shift_end * 60 > tw_start + SERVICE_TIME):
            candidates.append(len(route) - 1)

        if travel_time_based:

            times_added = []
            address = self.get_address(order.name)
            for index in candidates:
                pre_add = self.get_address(route[index][0])
                post_add = self.get_address(
                    route[index + 1][0]) if index < len(route) - 1 else 0
                orig = self.dur_mat[pre_add][post_add]
                new_time = self.dur_mat[pre_add][address] + self.dur_mat[address][post_add]
                times_added.append(new_time - orig)
            candidates_times = sorted(
                zip(candidates, times_added),
                key=lambda x: x[1]
            )
        
            for index, time in candidates_times:
                res, route = self.try_insert(van, route.copy(), order, index)
                if res:
                    return time, route
                
            return 0, route

        for index in reversed(candidates):
            # start trying from the later stops to minimize disruption
            res, route = self.try_insert(van, route, order, index)
            if res:
                return 1, route

        return 0, route


    def reroute(
        self,
        from_route: List[Tuple],
        to_route: List[Tuple],
        to_van: Vehicle
    ) -> Tuple[List, List]:

        new_to = to_route.copy()
        new_from = from_route.copy()

        for idx in range(len(from_route) - 1, 0, -1):
            order = self.data.loc[new_from[idx][0]]
            res, new_to = self.insert(to_van, new_to, order)
            if res:
                new_from.pop(idx)
            if len(new_from) <= len(new_to) + 1:
                break

        return new_from, new_to


    def get_centre(self, route: List[Tuple]) -> Tuple[float, float]:
        """Calculates the geographical centroid of of the route"""
        
        lat = sum(self.data.loc[o]['latitude'] for o, _ in route) / len(route)
        lng = sum(self.data.loc[o]['longitude'] for o, _ in route) / len(route)
        return lat, lng

    def get_distance(
        self,
        coor1: Tuple[float, float], 
        coor2: Tuple[float, float]
    ):
        return ((coor1[0] - coor2[0]) ** 2 + (
            coor1[1] - coor2[1]) ** 2) ** (1/2)

    def get_travel_time(
        self,
        route: List[Tuple]
    ) -> float:
        """Calculates the average travel time per route"""
        
        if len(route) == 2:
            return 0  # The route serving only one order

        return sum(
            self.dur_mat[self.get_address(route[i][0])][self.get_address(route[i+1][0])]
            for i in range(1, len(route) - 1)
        ) / (len(route) - 2)

    def get_travel_time_contribution(
        self,
        route: List[Tuple],
        index: int
    ):
        if index == 0:
            return 0
        
        curr = route[index][0]
        pre = route[index - 1][0]
        if index == len(route) - 1:
            post = 0
        else:
            post = route[index + 1][0]
        
        curr_add = self.get_address(curr)
        pre_add, post_add = self.get_address(pre), self.get_address(post)
        
        current = self.dur_mat[pre_add][curr_add] + self.dur_mat[curr_add][post_add]
        without = self.dur_mat[pre_add][post_add]
        return current - without

    def distance_based_balance(self, routes: Dict):
        
        total = sum(len(route[1:]) for _, route in routes.items())
        vans = sorted([
            van for van in self.meta['vans']
        ], key=lambda x: len(routes[x]), reverse=True)
        
        while True:
            
            # Identify longest route
            from_van = vans[0]
            from_route = routes[from_van]
            from_centre = self.get_centre(from_route[1:])
            
            # Calculate how far each order is from the centroid
            orders = [order for order, _ in from_route[1:]]
            coors = [
                (row['latitude'], row['longitude']) 
                for _, row in self.data.loc[orders].iterrows()
            ]
            distances = [
                self.get_distance(from_centre, coor)
                for coor in coors
            ]
            from_orders = sorted(
                zip(orders, distances, coors), 
                key=lambda x: x[1], reverse=True
            ) # from farthest to closest
            
            moved = False    
            
            # Try moving the farthest order to a different route
            for order, _, coor in from_orders:

                centres = [self.get_centre(routes[van][1:]) for van in vans]
                distances = [self.get_distance(coor, c) for c in centres]
                to_vans = sorted(zip(vans, distances), key=lambda x: x[1])
                
                # Try moving it to the closest cluster
                for to_van, _ in to_vans:
                    if from_van == to_van:
                        break
                    res, routes[to_van] = self.insert(
                        self.meta['vans'][to_van], 
                        routes[to_van], 
                        self.data.loc[order]
                    )
                    if res:
                        idx = [x for x, _ in routes[from_van]].index(order)
                        routes[from_van].pop(idx)
                        moved = True
                        break
                        
                if len(routes[from_van]) < total / len(vans):
                    break
                        
            vans.sort(key=lambda x: len(routes[x]), reverse=True)
            
            if not moved:
                break

    def naive_balance(self, routes: Dict):

        vans = sorted([
            van for van in self.meta['vans']
        ], key=lambda x: len(routes[x]), reverse=True)
        lengths = [len(routes[van]) for van in vans]
        new_var = variance(lengths)
        var = new_var + 1

        while new_var < var:

            var = new_var
            from_v = vans[0]
            for to_v in reversed(vans[1:]):
                routes[from_v], routes[to_v] = self.reroute(
                    routes[from_v], routes[to_v], self.meta['vans'][to_v])
            vans.sort(key=lambda x: len(routes[x]), reverse=True)
            lengths = [len(routes[van]) for van in vans]
            new_var = variance(lengths)

    def minimize_average_travel_time(self, routes: Dict):
    
        spreads = dict(
            (order, self.get_travel_time(route)) for order, route in routes.items())
        vans = sorted([
            van for van in self.meta['vans']
        ], key=lambda x: spreads[x], reverse=True)
        
        moved = False

        while True:
            
            from_van = vans[0]
            from_route = routes[from_van]
            
            # Calculate how much travel time each order contributes
            orders = [order for order, _ in from_route[1:]]
            contributions = [
                self.get_travel_time_contribution(from_route, index)
                for index in range(1, len(from_route))
            ]
            from_orders = sorted(
                zip(orders, contributions), 
                key=lambda x: x[1], reverse=True
            )

            for order, contribution in from_orders:
                
                # Try moving it to the closest cluster
                min_time_added = 100
                min_van = from_van
                new_routes = {}
                
                for to_van in vans[1:]:
                    res, new_routes[to_van] = self.insert(
                        self.meta['vans'][to_van], routes[to_van], self.data.loc[order], True)
                    if res and res < min_time_added:
                        min_time_added = res
                        min_van = to_van

                if new_routes and res < contribution:
                    routes[min_van] = new_routes[min_van]
                    idx = [x for x, _ in routes[from_van]].index(order)
                    routes[from_van].pop(idx)
                    break

            spreads = dict(
                (van, self.get_travel_time(route)) 
                for van, route in routes.items()
            )
            vans.sort(key=lambda x: spreads[x], reverse=True)

            if not moved:
                break

    def get_address(self, index: int) -> str:
        """Gets address based on address ID"""

        if index == 0:
            return self.meta['store_address_id']
        return self.data.loc[index]['address_id']
