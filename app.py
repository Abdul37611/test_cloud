from cmath import nan
import pandas as pd
from vrp import preprocess, cluster, tsp
from vrp.tsp import TSP
import streamlit as st
from jinja2 import Template
import gmaps
from IPython.display import display
from ipywidgets import embed
import streamlit.components.v1 as components
from flask import Flask
import os
from collections import defaultdict
import folium
import requests
import json
import polyline
import time



gmaps.configure(api_key="AIzaSyBLD0WOO-DmYYNisl-Qgku514-v7AgZz1c")
st.set_page_config(layout="wide")

with open("assets/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



#st.sidebar.image("assets/FullColor_1280x1024_300dpi.png", use_column_width=True)


# UI

uploaded_data=st.sidebar.file_uploader(label="Data File")
uploaded_config=st.sidebar.file_uploader(label="Config File")



if "run" not in st.session_state:
    st.session_state["run"]=False

if "first_run_ui" not in st.session_state:
    st.session_state["first_run_ui"]=True

if "first_run" not in st.session_state:
    st.session_state["first_run"]=True

if None not in (uploaded_data,uploaded_config):


    if st.session_state["first_run_ui"]==True:
        st.session_state["df_ori"]=pd.read_excel(uploaded_data,engine='openpyxl') # For getting addres and displaying in table 
        st.session_state["cf_ori"]=pd.read_excel(uploaded_config,engine='openpyxl') # For getting store information 
        st.session_state["cf_ori"] = st.session_state["cf_ori"].dropna(axis = 0, how = 'all')

        store_code=st.session_state["cf_ori"]["Store"].unique()
        store_address=st.session_state["cf_ori"]["Address"].unique()
        store_code=list(map(int,store_code))

        st.session_state["store_code"]=store_code
        st.session_state["store_address"]=store_address
        st.session_state["first_run_ui"]=False


    store = st.sidebar.selectbox("Store", st.session_state["store_code"])


    date = st.sidebar.date_input("Date",value=None)
    date=str(date)

    button_layout = st.sidebar.columns([1,1])
    with button_layout[0]:
        run=st.button("Data Preprocess")

    if run:
        st.session_state["run"]=True

        if "date" not in st.session_state:
            st.session_state["date"]=date

        if "store" not in st.session_state:
            st.session_state["store"]=store

        

else:
    st.sidebar.write("Please upload data file and config file")





if "date" in st.session_state:
    if date != st.session_state['date']:
            st.session_state["first_run"]=True
            st.session_state["date"]=date
            st.session_state["run"]=False
            st.session_state["run_algo"]=False

if "store" in st.session_state:
    if store != st.session_state['store']:
        st.session_state["first_run"]=True
        st.session_state["store"]=store
        st.session_state["run"]=False
        st.session_state["run_algo"]=False


if st.session_state["run"]==True:

    try:
        if os.path.exists('resources_test'):
            pass
        else:
            os.mkdir("resources_test")

        if os.path.exists(f"resources_test/{store}-{date}"):
            pass
        else:
            os.mkdir(f"resources_test/{store}-{date}")

    
        if st.session_state["first_run"] == True:
            
            with st.spinner(f'Preprocessing the data ...'):
                if ((os.path.exists(f'resources_test/{store}-{date}/processed_data.csv')) and (os.path.exists(f'resources_test/{store}-{date}/meta.json'))):
                    
                    data, meta = preprocess.read_processed_data(
                    f'resources_test/{store}-{date}/processed_data.csv', 
                    f'resources_test/{store}-{date}/meta.json'
                    )

                    st.session_state["data"]=data
                    st.session_state["meta"]=meta
                
                else:
                    data, config = preprocess.read_data(uploaded_data, uploaded_config)

                    output = preprocess.select_data(data, config, str(date) , str(store))
                   
                    if type(output) == list:
                        for i in output:
                            st.error(i)
                    else:
                        data, meta = output
                        
                        st.session_state["data"]=data
                        st.session_state["meta"]=meta
                        

                        with open(f"resources_test/{store}-{date}/meta.json", 'w') as fp:
                            pass

                        preprocess.save_data(data, meta, 
                        f'resources_test/{store}-{date}/processed_data.csv', 
                        f'resources_test/{store}-{date}/meta.json'
                        )

            with st.spinner(f'Getting the distance matrix ...'):
                if ((os.path.exists(f'resources_test/{store}-{date}/distance_matrix.csv')) and (os.path.exists(f'resources_test/{store}-{date}/duration_matrix.csv'))):
                
                    dist_mat = pd.read_csv(
                        f'resources_test/{store}-{date}/distance_matrix.csv',
                        index_col=0
                    )
                    dur_mat = pd.read_csv(
                        f'resources_test/{store}-{date}/duration_matrix.csv',
                        index_col=0
                    )

                    st.session_state["dist_mat"]=dist_mat
                    st.session_state["dur_mat"]=dur_mat

                else:
                    dist_mat, dur_mat = preprocess.get_dist_mat(
                        data, meta, 
                        f'resources_test/{store}-{date}/distance_matrix.csv',
                        f'resources_test/{store}-{date}/duration_matrix.csv'
                    )

                    st.session_state["dist_mat"]=dist_mat
                    st.session_state["dur_mat"]=dur_mat


            st.session_state["first_run"]=False

        if "run_algo" not in st.session_state:
            st.session_state["run_algo"]=False
        
        with button_layout[-1]:
            run_algo=st.button("Run Algorithm")
        
        if run_algo:
            st.session_state["run_algo"]=True
        

        if st.session_state["run_algo"]==True:

            if f"clustered_data_{store}_{date}" not in st.session_state:

                with st.spinner(f'Clustering the dataset ...'):
                    if os.path.exists(f'resources_test/{store}-{date}/clustered_data.csv'):
                        
                        clustered_data = pd.read_csv(
                            f'resources_test/{store}-{date}/clustered_data.csv',
                            index_col=0
                        )

                        st.session_state[f"clustered_data_{store}_{date}"]=clustered_data
                        st.session_state[f"tsp_solver_{store}_{date}"]=TSP(st.session_state[f"clustered_data_{store}_{date}"], st.session_state["dur_mat"], st.session_state["meta"])
                    else:

                        clustered_data = cluster.two_stage_cluster(st.session_state["data"], st.session_state["dur_mat"], st.session_state["meta"])

                        clustered_data.to_csv(f'resources_test/{store}-{date}/clustered_data.csv')

          
          
                        st.session_state[f"clustered_data_{store}_{date}"]=clustered_data
                        st.session_state[f"tsp_solver_{store}_{date}"]=TSP(st.session_state[f"clustered_data_{store}_{date}"], st.session_state["dur_mat"], st.session_state["meta"])

                
                    clusters=st.session_state[f"clustered_data_{store}_{date}"]["cluster"].unique()
                    clusters=sorted(clusters)


                    vehicles_shift={}
                    for i in clusters:
                        temp=i.rsplit(" ",1)
                        vhe=temp[0]
                        sft=temp[1]
                        sft=sft[:2]+":"+sft[2:]
                        
                        if vhe in vehicles_shift:
                            vehicles_shift[vhe].append(sft)
                        else:
                            vehicles_shift[vhe]=[sft]

                    veh_list=list(vehicles_shift.keys())
                    

                    st.session_state[f"veh_list_{store}_{date}"]=veh_list
                    st.session_state[f"vehicles_shift_{store}_{date}"]=vehicles_shift

            col1, col2 = st.columns(2) 

            with col1:
                vehicle=st.selectbox("Vehicle", st.session_state[f"veh_list_{store}_{date}"])

            with col2:
                shift=st.selectbox("Shift", st.session_state[f"vehicles_shift_{store}_{date}"][vehicle])
                shift=shift[:2]+shift[3:]
            
            if f"result_{store}_{date}_{vehicle}_{shift}" not in st.session_state:
                with st.spinner(f'Finding optimal route ...'):
                    st.session_state[f"result_{store}_{date}_{vehicle}_{shift}"] = st.session_state[f"tsp_solver_{store}_{date}"].solve(vehicle+" "+shift)

            

            if f"df_{store}_{date}_{vehicle}_{shift}" not in st.session_state:
                with st.spinner(f'Formatting the results in tabular form ...'):
                
                    result=st.session_state[f"result_{store}_{date}_{vehicle}_{shift}"]

                    result.pop(0)
                    
                    df=pd.DataFrame(result)

                    df[1]=pd.to_datetime(df[1], unit='m').dt.strftime('%H:%M')

                    df.columns = ['Order ID', 'Arival Time']
                    df["Address"]=1
                    df["Additional Info"]=""

                    for i in df['Order ID']:

                        if st.session_state["df_ori"].isin([i]).any().any():
                            index=st.session_state["df_ori"][st.session_state["df_ori"]['ORDER_NO']==i].index.values
                            
                            temp_list=list(st.session_state["df_ori"].iloc[index[0]])
                           
                            Address=""
                            Address+=temp_list[2].strip()
                            Address+=", "
                            Address+=temp_list[4].strip()
                            Address+=", "
                            Address+=temp_list[5].strip()
                            Address+=", "
                            Address+=temp_list[6].strip()
                            
                            df_index=df[df['Order ID']==i].index.values

                            df.iloc[df_index,2]=Address

                            df.iloc[df_index,3]= " " if type(temp_list[3]) == float else temp_list[3]
                            
                        
                            
                    st.session_state[f"df_{store}_{date}_{vehicle}_{shift}"]=df
                    


            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(st.session_state[f"df_{store}_{date}_{vehicle}_{shift}"])

            if f"map_{store}_{date}_{vehicle}_{shift}" not in st.session_state:
                with st.spinner(f'Ploting the optimal route ...'):
                    
                    store_coor=st.session_state["meta"]['store_coor']
                    


                    order_id = st.session_state[f"df_{store}_{date}_{vehicle}_{shift}"]["Order ID"]
                    waypoints=[]
                    for i in order_id:
                        waypoints.append((st.session_state[f"clustered_data_{store}_{date}"].loc[i][["latitude","longitude"]].tolist()))
                    

                    waypoints_1d=[]
                    for i in waypoints:
                        waypoints_1d.extend(i)

                    waypoints_string_osrm=[]
                    waypoints_string=[]
                    for i in range(len(waypoints_1d)):
                        if i+1 < len(waypoints_1d):
                            if i%2==0:
                                j=i+1
                                waypoints_string.append(f"{waypoints_1d[i]},{waypoints_1d[j]}")
                                waypoints_string_osrm.append(f"{waypoints_1d[j]},{waypoints_1d[i]}")
                    

                    def list_duplicates(seq):
                        tally = defaultdict(list)
                        for i,item in enumerate(seq):
                            tally[item].append(i)
                        return ((key,locs) for key,locs in tally.items() 
                                                if len(locs)>1)

                    dup_list=[]
                    for dup in sorted(list_duplicates(waypoints_string)):
                        dup_list.append(dup)

                    
                    for i in range(len(dup_list)):
                        lent=len(dup_list[i][1]) - 1
                        inc=0.0003
                        for j in range(lent):
                            coor=dup_list[i][0].split(",")
                            coor=list(map(float,coor))
                            coor[0]=coor[0]+inc
                            coor[1]=coor[1]+inc
                            ind=dup_list[i][1][j]
                            waypoints_string[ind]=f"{coor[0]},{coor[1]}"
                            inc+=0.0003
                    

                    
                    waypoints_final=[]
                    for i in waypoints_string:
                        a,b=eval(i)
                        waypoints_final.append({'lat': a, 'lng': b})

                    

                    waypoints_final.insert(0,{'lat':store_coor[0],'lng':store_coor[1]})
                    waypoints_final.append({'lat':store_coor[0],'lng':store_coor[1]})

                    
                    def render_map(waypoints):
                        with open("templates/stack_map_tst.html") as map:
                            template = Template(map.read())
                            return template.render(
                                waypoints=waypoints  
                            )
                    
                    rm=render_map(waypoints_final)
                    

                    waypoints_string_osrm.insert(0,f"{store_coor[1]},{store_coor[0]}")
                    waypoints_string_osrm.insert(-1,f"{store_coor[1]},{store_coor[0]}")

                    waypoints_final_osrm=";".join(waypoints_string_osrm)
                    
                    route_url=f'http://router.project-osrm.org/route/v1/driving/{waypoints_final_osrm}?alternatives=true&geometries=polyline'
                    r=requests.get(route_url)
                    res=r.json()

                    res_dist_dur=[]
                    
                    for i in range(len(res['routes'][0]['legs'])):
                        x=res['routes'][0]['legs'][i]['distance']
                        y=res['routes'][0]['legs'][i]['duration']

                        res_dist_dur.append([int(x)/1000,round(int(y)/60)])



                    st.session_state[f"res_{store}_{date}_{vehicle}_{shift}"]=res_dist_dur
                    st.session_state[f"map_{store}_{date}_{vehicle}_{shift}"]=rm


            

            if f"route_seg_{store}_{date}_{vehicle}_{shift}" not in st.session_state:
                with st.spinner(f'Getting route summary ...'):
                    
                    fianl_address_list=list(st.session_state[f"df_{store}_{date}_{vehicle}_{shift}"]['Address'])

                    num_customer=len(fianl_address_list)
                    

                    ind=st.session_state["store_code"].index(store)
                    fianl_address_list.append(st.session_state['store_address'][ind])
                    fianl_address_list.insert(0,st.session_state['store_address'][ind]) 
                    
                    route_segment_label=[]
                    for i in range(num_customer+1):
                        if i+1<num_customer+1:
                            j=i+1
                            
                            route_segment_label.append(f"{i} --> {j}")
                    
                    route_segment_label.pop(0)
                    route_segment_label.insert(0,"Depot --> 1")
                    route_segment_label.append(f"{num_customer} --> Depot")
                    


                    fianl_route_segment=[]
                    for i in range(len(fianl_address_list)):
                        if i+1 < len(fianl_address_list):
                            j=i+1
                            fianl_route_segment.append([f"{fianl_address_list[i]} --> {fianl_address_list[j]}"])
                    
                    for i in range(len(fianl_route_segment)):
                        fianl_route_segment[i].extend(st.session_state[f"res_{store}_{date}_{vehicle}_{shift}"][i])
                        fianl_route_segment[i].append(route_segment_label[i])

                    st.session_state[f"route_seg_{store}_{date}_{vehicle}_{shift}"]=fianl_route_segment


            

            def render_route_segment(rslist):
                with open("templates/r_seg.html") as cost:
                    template = Template(cost.read())
                    return template.render(
                        rslist=rslist
                )

            def render_solution_cost(dflist):
                with open("templates/df_table.html") as cost:
                    template = Template(cost.read())
                    return template.render(
                        dflist=dflist
                )

            dflist=st.session_state[f"df_{store}_{date}_{vehicle}_{shift}"].values.tolist()

            for i in range(len(dflist)):
                dflist[i][0]=int(dflist[i][0])
            
            rdf = render_solution_cost(dflist=dflist)

            px_height=(len(dflist)* 50) + 20 + 20

            rrs = render_route_segment(rslist=st.session_state[f"route_seg_{store}_{date}_{vehicle}_{shift}"])



            components.html(st.session_state[f"map_{store}_{date}_{vehicle}_{shift}"],height=750) #,height=750,width=1415
            components.html(rrs,height=750)

            components.html(rdf,height=px_height)

            st.download_button(
                "Press to Download",
                csv,
                "file.csv",
                "text/csv",
                key='download-csv'
                )

        else:
            pass

    except KeyError as k:
        st.write("Data for the particular store or date not found.")
        

