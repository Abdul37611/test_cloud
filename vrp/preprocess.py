import json
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Tuple

import googlemaps
import numpy as np
import pandas as pd

from .config import API_KEY

STORES = {
    '3165': '5400 Rue Jean-talon O, Montreal, QC, H4P 2T5',
    '3174': '670 Applewood Crescent, Vaughan, ON, L4K 4B4',
    '1004': '2525 St Clair Ave W, Toronto, ON, M6N 4Z5',
    '1056': '680 Laval Dr, Oshawa, ON, L1J 0B5',
    '1001': '1525 Bristol Road West Mississauga ON L5V 2P3',
    '2892': '17 Tannery Street Mississauga ON L5M 4Z0'
}


class Vehicle(NamedTuple):
    name: str
    store: int
    speed: float
    shift_start: int
    shift_end: int
    slots_served: List[int]


def read_data(
    data_file='dataset.xlsx',
    config_file='config.xlsx'
) -> Tuple[pd.DataFrame, Dict]:

    data = pd.read_excel(
        data_file,
        sheet_name='Export Worksheet',
        usecols=[
            'ORDER_NO',
            'ADDRESS_LINE1',
            'ADDRESS_LINE2',
            'CITY',
            'ZIP_CODE',
            'EXTN_MTEP_SHIP_NODE',
            'EXTN_MTEP_DEL_START_DATE_TIME',
            'EXTN_MTEP_DEL_END_DATE_TIME',
            'EXTN_MTEP_DISPENSE_TYPE',
            'EXTN_MTEP_VAN_ID'
        ],
        parse_dates=[
            'EXTN_MTEP_DEL_START_DATE_TIME',
            'EXTN_MTEP_DEL_END_DATE_TIME'
        ],
        dtype={
            'EXTN_MTEP_SHIP_NODE': str
        }
    ,engine='openpyxl').rename(
        columns={
            'ORDER_NO': 'order_id',
            'ADDRESS_LINE1': 'address',
            'ADDRESS_LINE2': 'address_line2',
            'CITY': 'city',
            'ZIP_CODE': 'zip',
            'EXTN_MTEP_SHIP_NODE': 'store',
            'EXTN_MTEP_DEL_START_DATE_TIME': 'start_datetime',
            'EXTN_MTEP_DEL_END_DATE_TIME': 'end_datetime',
            'EXTN_MTEP_DISPENSE_TYPE': 'dispense_type',
            'EXTN_MTEP_VAN_ID': 'van_id'
        }
    )

    conf_df = pd.read_excel(
        config_file,
        sheet_name='Vehicles&Shift',
        usecols=[
            'Vehicle',
            'Store',
            'Vehicle Speed',
            'Shift Start',
            'Shift End',
            'Break Start',
            'Break End',
            'Slots Served',
            'Days Active',
            'Distance Threshold (all orders exceeding this travel distance from the store MUST be taken by a truck)'
        ],
        dtype={
            'Store': str,
            'Shift Start': str,
            'Shift End': str,
            'Break Start': str,
            'Break End': str,
        }
    ,engine='openpyxl').rename(columns={
        'Distance Threshold (all orders exceeding this travel distance from the store MUST be taken by a truck)': 'Distance Threshold'
    })

    conf_df['Vehicle'] = conf_df['Vehicle'] + ' ' + conf_df['Shift Start']
    
    conf_df = conf_df.dropna(axis = 0, how = 'all')

    config = {}
    for store in pd.unique(conf_df['Store']):
        vehicles: Dict[str, Vehicle] = {}
        store_df = conf_df.loc[conf_df['Store'] == store]
        for _, row in store_df.iterrows():
            name = row['Vehicle'][:-5]
            speed = row['Vehicle Speed']
            shift_start = int(row['Shift Start'][:-2])
            shift_end = int(row['Shift End'][:-2])
            slots_served = [
                int(slot.split('-')[0])
                for slot in row['Slots Served'].split(',')
            ]
            vehicles[row['Vehicle']] = Vehicle(
                name, store, speed, shift_start, shift_end, slots_served)
        threshold = int(store_df.iloc[0]['Distance Threshold'][:-2])

        config[store] = {
            'vehicles': vehicles,
            'distance_threshold': threshold
        }

    return data, config


def select_data(
    data: pd.DataFrame,
    config: Dict,
    date: str,
    store: str
) -> Tuple[pd.DataFrame, Dict]:
    """Select data for one store at a particular date

    The date should be in the form of '2022-02-01', and the store should be one
    of 3165, 3174, 1004, 1056.

    """

    meta = {}

    # Select data from the requested date
    next_day = (datetime.strptime(date, '%Y-%m-%d') + timedelta(1))
    next_day_str = next_day.strftime('%Y-%m-%d')
    data = data.loc[
        (data['start_datetime'] > date) &
        (data['start_datetime'] < next_day_str)
    ]

    # Filter data by store
    data = data.loc[data['store'] == store]
    store_address = STORES[store]

    # Process time windows
    data['tw_start'] = data['start_datetime'].apply(lambda x: x.hour)
    data['tw_end'] = data['end_datetime'].apply(lambda x: x.hour)

    # Vehicles
    vans = config[store]['vehicles']

    # Save info
    meta['vans'] = vans
    meta['distance_threshold'] = config[store]['distance_threshold']
    meta['store_address'] = store_address
    meta['store_id'] = store

    gmaps = googlemaps.Client(key=API_KEY)

    # Clean up address fields
    data['city'] = data['city'].apply(lambda x: x.upper().strip())
    data['zip'] = data['zip'].str[:3] + ' ' + data['zip'].str[-3:]
    addresses = data['address'] + ', ' + data['city'] + ' ' + data['zip']

    # Find addresses on Google Map
    store_address_id = gmaps.find_place(
        store_address, 'textquery')['candidates'][0]['place_id']

    if store == "1001":
        store_address_id= 'ChIJg6KoFqlBK4gREonlbBcdKGk'
    elif store == "2892":
        store_address_id= 'ChIJ5xzrH7lBK4gRmeDvfmAcKNY'
    else:
        store_address_id = gmaps.find_place(store_address, 'textquery')['candidates'][0]['place_id']

    
    res = addresses.apply(lambda x: gmaps.find_place(
        x, 'textquery')).apply(pd.Series)

    # Check validity of addresses
    errors = data[res['status']=='ZERO_RESULTS']
    address_error=[]
    for line, err in errors.iterrows():
        address = err['address']
        address_error.append(f'Address "{address}" on line {line} not recognized! Please fix '
            'the address in the data set, remember to remove additional text '
            'such as buzzer codes as it is not recognized by Google Maps.')
    
    if len(address_error)!=0:
        return address_error
        
    data['address_id'] = res['candidates'].apply(lambda x: x[0]['place_id'])

    # Clean up address entries in table
    address_info = data.address_id.apply(
        lambda _id: gmaps.place(_id)['result'])
    data['full_address'] = address_info.apply(
        lambda info: info['formatted_address'])
    data['longitude'] = address_info.apply(
        lambda info: info['geometry']['location']['lng'])
    data['latitude'] = address_info.apply(
        lambda info: info['geometry']['location']['lat'])

    # Save store info
    meta['store_address_id'] = store_address_id
    store_info = gmaps.place(store_address_id)['result']
    meta['store_coor'] = [
        store_info['geometry']['location']['lat'],
        store_info['geometry']['location']['lng']
    ]

    data['distance'] = data['address_id'].apply(
        lambda x: get_shortest_distance(x, meta['store_address_id'], gmaps))

    return data, meta


def get_dist_mat(
    data: pd.DataFrame,
    meta: Dict,
    dist_mat_file: str,
    dur_mat_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Find the distance matrix"""

    gmaps = googlemaps.Client(key=API_KEY)

    store_address_id = 'place_id:' + meta['store_address_id']
    addresses = [store_address_id] + [
        'place_id:' + x for x in pd.unique(data.address_id).tolist()]

    n_addresses = len(addresses)
    dist_mat = pd.DataFrame()
    dur_mat = pd.DataFrame()
    for i in range(int(np.ceil(n_addresses / 10))):
        sub_dist_row = pd.DataFrame()
        sub_dur_row = pd.DataFrame()
        for j in range(int(np.ceil(n_addresses / 10))):
            response = gmaps.distance_matrix(
                origins=addresses[i*10:i*10+10],
                destinations=addresses[j*10:j*10+10]
            )
            rows = [row['elements'] for row in response['rows']]
            sub_dist_mat = pd.DataFrame([[
                int(np.ceil(e['distance']['value'])) for e in row
            ] for row in rows])
            sub_dur_mat = pd.DataFrame([[
                int(np.ceil(e['duration']['value'] / 60)) for e in row
            ] for row in rows])  # to the whole minute
            sub_dist_row = pd.concat([sub_dist_row, sub_dist_mat], axis=1)
            sub_dur_row = pd.concat([sub_dur_row, sub_dur_mat], axis=1)
        dur_mat = pd.concat([dur_mat, sub_dur_row], axis=0)
        dist_mat = pd.concat([dist_mat, sub_dist_row], axis=0)

    address_ids = [x[9:] for x in addresses]
    dist_mat.columns = address_ids
    dist_mat['index'] = address_ids
    dist_mat = dist_mat.set_index('index')
    dist_mat.index.name = None
    dur_mat.columns = address_ids
    dur_mat['index'] = address_ids
    dur_mat = dur_mat.set_index('index')
    dur_mat.index.name = None

    dist_mat.to_csv(dist_mat_file)
    dur_mat.to_csv(dur_mat_file)

    return dist_mat, dur_mat


def save_data(data: pd.DataFrame, meta: Dict, data_file: str, meta_file: str):

    data.to_csv(data_file, columns=[
        'order_id',
        'address',
        'address_line2',
        'city',
        'zip',
        'longitude',
        'latitude',
        'address_id',
        'store',
        'tw_start',
        'tw_end',
        'start_datetime',
        'end_datetime',
        'distance',
        'dispense_type',
        'van_id']
    )

    with open(meta_file, 'w') as outfile:
        json.dump(json.dumps(meta), outfile)


def save_config(config: Dict, filename: str):
    with open(filename, 'w') as outfile:
        json.dump(json.dumps(config), outfile)


def read_processed_data(
    data_file: str,
    meta_file: str
) -> Tuple[pd.DataFrame, Dict]:

    data = pd.read_csv(
        data_file,
        index_col=0,
        parse_dates=[
            'start_datetime',
            'end_datetime'
        ],
        dtype={
            'store': str
        }
    )

    with open(meta_file, 'r') as f:
        meta = json.loads(json.load(f))

    for vehicle in meta['vans']:
        meta['vans'][vehicle] = Vehicle(*meta['vans'][vehicle])

    return data, meta


def read_config(config_file: str) -> Dict:

    with open(config_file, 'r') as f:
        config = json.loads(json.load(f))

    for store in config:
        for vehicle in config[store]['vehicles']:
            config[store]['vehicles'][vehicle] = Vehicle(
                *config[store]['vehicles'][vehicle])

    return config


def get_shortest_distance(
    address_id: str, 
    store_address: str, 
    gmaps
) -> int:
    result = gmaps.directions(
        'place_id:' + store_address, 
        'place_id:' + address_id, 
        alternatives=True
    )
    return min(alt['legs'][0]['distance']['value'] for alt in result)
