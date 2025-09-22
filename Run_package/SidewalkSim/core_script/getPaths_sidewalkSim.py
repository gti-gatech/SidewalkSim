import os
import glob
from itertools import islice

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import networkx as nx
import time
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')
# os.environ['PROJ_LIB'] = '/Volumes/Samsung_T5/Courses/CEE6701_planning/Scripts/02_SidewalkSim'  # need to change this when necessary


def initialize_abm15_links():  # needed only when updating the abm15 network
    """
    Import shape-file to GeoDataFrame, convert the format based on our needs.
    (Combining all useful information from 2 .shp file to one file)
    Store the output file back to .shp file and return the DataFrame.
    After prepared, output .shp file could be used the next time and
    there is no need to call this function again.

    :return: GeoDataFrame of network link information, along with its end node information.
    """
    # edit file directory
    file_dir = os.path.join(os.environ['PROJ_LIB'], 'ABM2020 203K')
    file_name_nodes = os.path.join('2020 nodes with latlon', '2020_nodes_latlon.shp')
    file_name_links = os.path.join('2020 links', '2020_links.shp')

    df_nodes_raw = gpd.read_file(os.path.join(file_dir, file_name_nodes))
    df_links_raw = gpd.read_file(os.path.join(file_dir, file_name_links))
    df_nodes = df_nodes_raw[['N', 'X', 'Y', 'lat', 'lon']]
    df_links_raw = df_links_raw[df_links_raw['FACTYPE'] > 0][df_links_raw['FACTYPE'] < 50]
    if 'SPEEDLIMIT' in df_links_raw.columns.tolist():
        spd = 'SPEEDLIMIT'
    else:
        spd = 'SPEED_LIMI'

    if 'A_B' not in df_links_raw.columns.tolist():
        df_links_raw['A_B'] = df_links_raw.apply(lambda x: str(int(x['A'])) + '_' + str(int(x['B'])), axis=1)

    mapper = {0: 35, 3: 65, 4: 65, 7: 35, 10: 35, 11: 70, 14: 35}  # edited hl 04092019
    df_links_raw['tmp'] = df_links_raw['FACTYPE'].map(mapper)
    df_links_raw['tmp'] = df_links_raw['tmp'].fillna(35)
    df_links_raw.loc[df_links_raw[spd] == 0, spd] = df_links_raw.loc[df_links_raw[spd] == 0, 'tmp']
    df_links = df_links_raw[['A', 'B', 'A_B', 'geometry', spd, 'DISTANCE', 'NAME']]
    df_links = df_links.merge(df_nodes.rename(columns={'N': 'A', 'X': 'Ax', 'Y': 'Ay', 'lat': 'A_lat', 'lon': 'A_lon'}),
                              how='left', on='A')
    df_links = df_links.merge(df_nodes.rename(columns={'N': 'B', 'X': 'Bx', 'Y': 'By', 'lat': 'B_lat', 'lon': 'B_lon'}),
                              how='left', on='B')

    def abm15_assignGrid(df_links, grid_size=25000.0):
        for col in ['minx', 'miny', 'maxx', 'maxy']:
            df_links[col + '_sq'] = round(df_links['geometry'].bounds[col] / grid_size, 0)
        # df_links['dist']=df_links['geometry'].length
        return df_links

    df_links = abm15_assignGrid(df_links)
    df_links = gpd.GeoDataFrame(df_links, geometry=df_links['geometry'], crs=df_links.crs)
    df_links.to_file(os.path.join(os.environ['PROJ_LIB'], 'build_graph',
                                  'data_node_link', 'abm15_links.shp'))
    return df_links


# init SidewalkSim links
def initialize_sws_links():
    """
    Similar to initialize_abm15_links formula, but prepare network for SidewalkSim.

    Add node information to link files.

    :return: GeoDataFrame of network link information, along with its end node information.
    """
    # step 1. import files
    file_dir = os.path.join(os.environ['PROJ_LIB'], 'sidewalk_raw_files')

    file_name_nodes = os.path.join(file_dir, 'sidewalkNodes.shp')
    file_name_sw = os.path.join(file_dir, 'sidewalks.shp')
    file_name_cw = os.path.join(file_dir, 'crosswalks.shp')

    df_nodes = gpd.read_file(file_name_nodes)
    df_links_sw = gpd.read_file(file_name_sw)
    df_links_cw = gpd.read_file(file_name_cw)

    # step 2. add geometry columns for node file
    df_nodes['X'] = df_nodes.geometry.x
    df_nodes['Y'] = df_nodes.geometry.y
    df_nodes = df_nodes.to_crs(epsg=4326)
    df_nodes['lon'] = df_nodes.geometry.x
    df_nodes['lat'] = df_nodes.geometry.y
    df_nodes = df_nodes[['sid', 'X', 'Y', 'lon', 'lat']]

    # step 3. merge two table into one by appending
    df_links_sw['type'] = 'sidewalk'
    df_links_cw['type'] = 'crosswalk'
    df_links = pd.concat([df_links_sw, df_links_cw])
    df_links = df_links.rename(columns={'sid1': 'A', 'sid2': 'B'})

    # step 3. merge node info to link file
    df_links = df_links.merge(
        df_nodes.rename(columns={'sid': 'A', 'X': 'Ax', 'Y': 'Ay', 'lat': 'A_lat', 'lon': 'A_lon'}),
        how='left', on='A')
    df_links = df_links.merge(
        df_nodes.rename(columns={'sid': 'B', 'X': 'Bx', 'Y': 'By', 'lat': 'B_lat', 'lon': 'B_lon'}),
        how='left', on='B')

    # step 4. add grid info for efficiency
    def abm15_assignGrid(df_links, grid_size=25000.0):
        for col in ['minx', 'miny', 'maxx', 'maxy']:
            df_links[col + '_sq'] = round(df_links['geometry'].bounds[col] / grid_size)
        # df_links['dist']=df_links['geometry'].length
        return df_links

    df_links = abm15_assignGrid(df_links)
    df_links = gpd.GeoDataFrame(df_links, geometry=df_links['geometry'], crs=df_links.crs)
    df_links['InterID_t'] = df_links['InterID_t'].fillna(-1)

    # step 5. get distance for every link; create name id as 'sid1_sid2'
    # (no direction contained sid1 is just the smaller id)
    df_links['distance'] = df_links['geometry'].length
    df_links['A_B'] = df_links['A'].astype(str) + '_' + df_links['B'].astype(str)

    # step 6. store computation results for reuse
    df_links.to_file(os.path.join(os.environ['PROJ_LIB'], 'build_graph',
                                  'data_node_link', 'SWS_links.shp'))
    return df_links


def build_bike_network(df_links):
    """
    Given original network like the DataFrame of abm15, create directed graph
        (bike network) for shortest paths searching using the package networkx.
    :param df_links: The whole network, like the whole abm15 geo-dataframe
    :return: DGo: directed link graph.
    """
    # TODO: maybe prepare bike speed in previous step instead of this simplification
    df_links['bk_speed'] = 15  # 15 mph
    # The measure to use as cost of traverse a link
    col = 'time'  # can be expand to other measures, like considering grades, etc.

    def compute_link_cost(x, method):
        x[method] = x['DISTANCE'] / x['bk_speed']  # mile / mph = hour
        # could implement other methods based on needs (e.g., consider grades, etc.)
        return x[method]

    df_links[col] = df_links.apply(compute_link_cost, axis=1, method=col)

    DGo = nx.DiGraph()  # directed graph
    for ind, row2 in df_links.iterrows():
        # forward graph, time stored as minutes
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[col]) * 60.0)],
                                    weight='forward', dist=row2['DISTANCE'], name=row2['NAME'])
    return DGo


def build_walk_network(df_links):
    """
    Given original network like the DataFrame of abm15, create directed graph
        (bike network) for shortest paths searching using the package networkx.
    :param links_dict: containing two GeoDataFrame sidewalk links and crosswalk links.
    :return: DGo: directed link graph.
    """
    df_links['walk_speed'] = 3  # 3 mph
    # The measure to use as cost of traverse a link
    col = 'time'  # can be expand to other measures, like considering grades, etc.

    def compute_link_cost(x, method):
        x[method] = (x['distance'] / 5280) / x['walk_speed']  # mile / mph = hour
        # could implement other methods based on needs (e.g., consider grades, etc.)
        return x[method]

    df_links[col] = df_links.apply(compute_link_cost, axis=1, method=col)

    DGo = nx.DiGraph()  # undirected graph
    # create forward graph, time stored as minutes
    for ind, row2 in df_links.iterrows():
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[col]) * 60.0)],
                                    weight='forward', dist=row2['distance'] / 5280, name=row2['InterID_t'])
        DGo.add_weighted_edges_from([(str(row2['B']), str(row2['A']), float(row2[col]) * 60.0)],
                                    weight='forward', dist=row2['distance'] / 5280, name=row2['InterID_t'])
    return DGo


# add (x,y) given (lon, lat)
def add_xy(df, lat, lon, x, y, x_sq, y_sq, grid_size=25000.0):
    """
    Given (lat, lon) information, generate coordinates in local projection system
        Also, classify location into different categories using a grid and store the
        row and column it falls into.
    """
    crs = {'init': 'epsg:4326', 'no_defs': True}  # NAD83: EPSG 4326
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    df = df.to_crs(epsg=2240)  # Georgia West (ftUS):  EPSG:2240
    df[x] = df['geometry'].apply(lambda x: x.coords[0][0])
    df[y] = df['geometry'].apply(lambda x: x.coords[0][1])
    df[x_sq] = round(df[x] / grid_size, 0)
    df[y_sq] = round(df[y] / grid_size, 0)
    return df


def build_walk_network_impedance(df_links, col):
    """
    Given original network like the DataFrame of abm15, create directed graph
        (bike network) for shortest paths searching using the package networkx.
    Arguments:
        df_links: GeoDataFrame network files like abm15.shp,
                  each row denotes a directed link with two end nodes A and B.
        col: impedance column, in travel time equivalence (seconds)
    :param links_dict: containing two GeoDataFrame sidewalk links and crosswalk links.
    :return: DGo: directed link graph.
    """

    DGo = nx.DiGraph()  # undirected graph
    # create forward graph, time stored as minutes
    for ind, row2 in df_links.iterrows():
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[col]) / 60.0)], # seconds to minutes
                                    weight='forward', dist=row2['distance'] / 5280, name=row2['InterID_t'])
        DGo.add_weighted_edges_from([(str(row2['B']), str(row2['A']), float(row2[col]) / 60.0)], # seconds to minutes
                                    weight='forward', dist=row2['distance'] / 5280, name=row2['InterID_t'])
    return DGo


def point_to_node(df_points, df_links, ifGrid=False, walk_speed=2.0, grid_size=25000.0, dist_thresh=5280.0):
    """
    Given a column of location projected to local coordinates (x, y), find nearest node in the network,
     record the node ID and the distance to walk to the node.
    Arguments:
        df_points: a DataFrame containing projected coordinates.
                   Each row corresponds to one point.
        df_links: GeoDataFrame network files like abm15.shp,
                  each row denotes a directed link with two end nodes A and B.
        ifGrid: If False, compute the grid it false into. If True, grid info is stored in de_points.
        walk_speed: walking speed default is 2.0 mph.
        grid_size: (I guess it should be) the width of the grid. Default is 25000 ft or 4.7 mile.
        dist_thresh: the maximum distance a normal person willing walk. Default is 1 mile.

    Returns:
        df_points: expand same input DataFrame with information about the nearest node and
                   walking time from point to the node.
    """

    def find_grid(pt_x):
        return round(pt_x / grid_size)

    def define_gridid(df_pts):
        df_pts['x_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][0]))
        df_pts['y_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][1]))
        return df_pts

    def find_closestLink(point, lines):
        dists = lines.distance(point)
        # print('dists shapes', dists.shape)
        return [dists.argmin(), dists.min()]

    def calculate_dist(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    # INITIALIZATION
    if ifGrid:
        df_points = define_gridid(df_points)
    df_points['NodeID'] = 0
    df_points['Node_t'] = 0
    # CALCULATION
    for ind, row in df_points.iterrows():
        try:
            # find links in the grid
            df_links_i = df_links[(df_links['minx_sq'] <= row['x_sq']) & \
                (df_links['maxx_sq'] >= row['x_sq']) & \
                (df_links['maxy_sq'] >= row['y_sq']) & \
                (df_links['miny_sq'] <= row['y_sq'])] ## Fizzy's edit, Sep. 2021
            # print('# of links in the grid', len(df_links_i))
            # print(df_links_i.index)
            # find the closest link and the distance
            LinkID_Dist = find_closestLink(row.geometry, gpd.GeoSeries(df_links_i.geometry))
            #print('closest', LinkID_Dist)
            linki = df_links_i.iloc[LinkID_Dist[0]] ## Fizzy's edit, Sep. 2021
            # find the closest node on the link
            df_coords = df_points.loc[ind, 'geometry'].coords[0]
            # print('coords', df_coords)
            dist1 = calculate_dist(df_coords[0], df_coords[1], linki['Ax'], linki['Ay'])
            dist2 = calculate_dist(df_coords[0], df_coords[1], linki['Bx'], linki['By'])
            if (dist1 > dist_thresh) and (dist2 > dist_thresh):
                df_points.loc[ind, 'NodeID'] = -1
                df_points.loc[ind, 'Node_t'] = -1
            else:
                df_points.loc[ind, 'NodeID'] = linki['A'] if dist1 < dist2 else linki['B']
                df_points.loc[ind, 'Node_t'] = dist1 / walk_speed / 5280.0 if \
                    dist1 < dist2 else dist2 / walk_speed / 5280.0
            # add distance o_d, d_d to dataframe
            df_points.loc[ind, 'dist'] = min(dist1, dist2) / 5280.0
        except Exception as e:
            print('Error happens!', e)
            assert False
            df_points.loc[ind, 'NodeID'] = -1
            df_points.loc[ind, 'Node_t'] = 0
    return df_points


def samp_pre_process(filename, dict_settings, option='bike'):

    df_links = dict_settings['network'][option]['links']
    walk_speed, grid_size, ntp_dist_thresh = dict_settings['walk_speed'], dict_settings['grid_size'], dict_settings[
        'ntp_dist_thresh']

    df_points = pd.read_csv(filename)

    # print(df_points)
    df_points = add_xy(df_points, 'ori_lat', 'ori_lon', 'x', 'y', 'x_sq', 'y_sq')
    df_points = point_to_node(df_points, df_links, False, walk_speed, grid_size, ntp_dist_thresh) \
        .rename(columns={'NodeID': 'o_node', 'Node_t': 'o_t', 'x': 'ox', 'y': 'oy',
                         'x_sq': 'ox_sq', 'y_sq': 'oy_sq', 'dist': 'o_d'})
    # print(df_points)
    df_points = add_xy(df_points, 'dest_lat', 'dest_lon', 'x', 'y', 'x_sq', 'y_sq')
    df_points = point_to_node(df_points, df_links, False, walk_speed, grid_size, ntp_dist_thresh) \
        .rename(columns={'NodeID': 'd_node', 'Node_t': 'd_t', 'x': 'dx', 'y': 'dy',
                         'x_sq': 'dx_sq', 'y_sq': 'dy_sq', 'dist': 'd_d'})
    # print(df_points)
    # origin, destination should be of different OD nodes and must found ODs to continue
    df_points = df_points[df_points['o_node'] != -1][df_points['d_node'] != -1][
        df_points['o_node'] != df_points['d_node']]
    # print(df_points.shape)
    if dict_settings['one_by_one'] is False:
        df_points.to_csv(filename.replace('.csv', '_node.csv').replace('samples_in', 'samples_out'), index=False)
    else:
        df_points.to_csv(filename.replace('.csv', '_node.csv').replace('one_by_one_in',
                                                                       'one_by_one_out'), index=False)
    return df_points


def format_routes(row, option, dict_settings, resultsPathi):
    """
    The only thing this function do it to format information
        correctly to a table.
    :param row: general information of a trip
    :param option: mode considered. now only has 'bike'
    :param dict_settings: settings, including the graph
    :param resultsPathi: one or several trip routes found given the trip info
    :return: a DataFrame with the correct record shape
    """
    # load network
    dict_bike = dict_settings['network'][option]
    DGo, links = dict_bike['DG'], dict_bike['links']
    # strategy 1 for forward, 2 for backward
    graph_type = ['forward', 'backward']
    strategy = dict_settings['strategy'][option]  # will be over-written if use 'one-by-one' mode
    graph_direction = graph_type[strategy - 1]
    # create DataFrame to hold data
    column_names = ['A', 'B', 'dist', 'mode',
                    'strategy', 'route_num', 'sequence',
                    'time', 'timeStamp', 'trip_id']
    # for each route, iterate paths in traveling order
    # A	B	dist	mode	option	route	sequence	time	timeStamp	trip_id	route_id
    returned_df = pd.DataFrame(columns=column_names)
    for routei, path in enumerate(resultsPathi):
        # initialize dataframe at the start of every loop
        if dict_settings['one_by_one'] is True:
            strategy = row['strategy']
            graph_direction = graph_type[strategy - 1]
        formated_df = pd.DataFrame(columns=column_names)
        print('PATH #:', routei)
        print(path)
        f_row = {}
        dists = 0
        accu_time = 0

        # attach first walking part before loop
        f_row['A'], f_row['B'], f_row['mode'] = 'origin', path[0], 'walk'
        f_row['strategy'], f_row['sequence'] = strategy, 1
        # need to demonstrate travel time in minutes for convenience
        f_row['time'], f_row['dist'] = row['o_t'] * 60, row['o_d']
        f_row = pd.DataFrame(f_row, index=[0])
        f_row['route_num'] = routei
        formated_df = pd.concat([formated_df, f_row], ignore_index=True)

        f_row = {}
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            f_row['A'], f_row['B'] = u, v
            dists += DGo[u][v]['forward']
            f_row['dist'] = DGo[u][v]['dist']
            f_row['mode'] = option
            f_row['route'] = DGo[u][v]['name']
            f_row['strategy'] = strategy
            f_row['sequence'] = i + 2

            f_row['time'] = DGo[u][v]['forward']  # the time it takes to travel on the link
            accu_time += f_row['time']

            f_row = pd.DataFrame(f_row, index=[0])
            f_row['route_num'] = routei
            formated_df = pd.concat([formated_df, f_row], ignore_index=True)

        # attach final walking part after loop
        f_row = {}
        f_row['A'], f_row['B'], f_row['mode'] = path[-1], 'destination', 'walk'
        f_row['strategy'], f_row['sequence'] = strategy, len(path) + 1
        # need to demonstrate travel time in minutes for convenience
        f_row['time'], f_row['dist'] = row['d_t'] * 60, row['d_d']
        f_row['route_num'] = routei
        f_row = pd.DataFrame(f_row, index=[0])

        formated_df = pd.concat([formated_df, f_row], ignore_index=True)

        # finally cumsum for each trip
        trip_num_rows = len(path) + 2
        if graph_direction == "forward":
            formated_df['timeStamp'] = formated_df['time'].cumsum()
        else:
            formated_df['timeStamp'] = formated_df['time'].iloc[::-1].cumsum() * -1
            formated_df['timeStamp'] = formated_df['timeStamp'] + formated_df['time']
        formated_df['timeStamp'] = formated_df['timeStamp'] / 60  # change to time stamp to hours from minutes
        if dict_settings['one_by_one'] is False:
            formated_df['timeStamp'] = formated_df['timeStamp'] + dict_settings['query_time']
        else:
            if strategy == 1:
                formated_df['timeStamp'] = formated_df['timeStamp'] + row['ori_time']
            else:  # strategy == 2
                formated_df['timeStamp'] = formated_df['timeStamp'] + row['dest_time']

        # update returned dataframe
        returned_df = pd.concat([returned_df, formated_df])

    # print(formated_df)
    returned_df['route_num'] = returned_df['route_num'] + 1
    return returned_df


def Bike_route_finder(row, option, dict_settings):
    """
    Parameters.
        row: a DataFrame row of travel information
            expected columns are: ox, oy, o_t, o_node, ox_sq, oy_sq, o_d
                                  dx, dy, d_t, d_node, dx_sq, dy_sq, d_d
        option: 'bike' for bike and walk only
        dict_settings: import all settings
    Returns.
        resultsPathi: path
        runningLogi: log information
    """
    t1 = time.time()

    # based on strategy, select forward or backward graph
    # strategy 1 for forward, 2 for backward
    # for BikewaySim, there is only forward case
    graph_type = ['forward', 'backward']
    if dict_settings['one_by_one'] is True:
        strategy = dict_settings['strategy'][option]
    else:
        strategy = dict_settings['strategy'][option]
    graph_direction = graph_type[strategy - 1]
    # load the number of k-shortest paths required
    num_routes = dict_settings['num_options'][option]

    # load network
    dict_bike = dict_settings['network'][option]
    DGo, links = dict_bike['DG'], dict_bike['links']

    # # Dijkstra search distance from origin to destination
    # print('**Dijkstra paths and distances**')
    # dists, paths = nx.single_source_dijkstra(DGo, str(row['o_node']), str(row['d_node']), weight=graph_direction)
    # print(paths)
    # print(dists)
    # print('**k-shortest paths results**')

    def k_shortest_paths(G, source, target, k, weight=None):
        if weight == "backward":
            reverse = True
            weight = "forward"  # no need to reset all name attributes of the network, just go with "forward"
        else:
            reverse = False
        if reverse:  # reverse the graph, search from target to source
            G = nx.DiGraph.reverse(G)
            path_generator = nx.shortest_simple_paths(G, source, target, weight)
        else:
            path_generator = nx.shortest_simple_paths(G, source, target, weight)
        paths = []
        i = 0
        while i < k:
            try:
                paths.append(next(path_generator))
                i += 1
            except:
                break
        # paths = list(islice(nx.shortest_simple_paths(G, source, target, weight), k))
        dists_lst = []
        for path in paths:
            dists = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                dists += G[u][v][weight]
            dists_lst.append(dists)

        if reverse:  # reverse back to the travel sequence order
            paths = [p[::-1] for p in paths]
            dists_lst = [d[::-1] for d in paths]
        return dists_lst, paths

    dists, paths = k_shortest_paths(DGo, str(row['o_node']), str(row['d_node']), k=num_routes, weight=graph_direction)
    if len(paths) == 0: ########## temporary solution
        dists = [nx.shortest_path_length(DGo, str(row['o_node']), str(row['d_node']))]
        paths = [nx.shortest_path(DGo, str(row['o_node']), str(row['d_node']))]

    # print(dists)
    # print(paths)

    # if need to plot, plot it!
    plot_all = dict_settings['plot_all']
    if 'limits' in dict_settings.keys():
        xlim = dict_settings['limits']['xlim']
        ylim = dict_settings['limits']['ylim']
    else:
        xlim = None
        ylim = None
    if plot_all:
        for i, path in enumerate(paths):
            if option == 'bike':
                trip_plot = plot_seq_abm(path, gpd_df=dict_settings['network'][option]['links'], xlim=xlim, ylim=ylim)
            if option == 'sidewalk':
                trip_plot = plot_seq_sw(path, gpd_df=dict_settings['network'][option]['links'], trip_series=row, xlim=xlim, ylim=ylim)
            file_name = "{}_{}.PNG".format(row['trip_id'], i)
            pn = os.path.join(dict_settings['plot_folder'], file_name)
            print(pn)
            trip_plot.savefig(pn)
            # trip_plot.show() # Fizz's Edit, Nov. 2021

    resultsPathi = format_routes(row, option, dict_settings, paths)
    err_message = 'Nothing wrong happens'
    numRoutes = len(dists)
    runningLogi = pd.DataFrame(
        {'trip_id': [row['trip_id']], 'option': ['drive'], 'state': [err_message], 'numRoutes': [numRoutes],
         'runTime': [time.time() - t1]})
    return resultsPathi, runningLogi


def BIGRun(row, options, dict_settings):
    # if O/D distancee are within walking threshold, there is no need to bike/drive/transit.
    def determine_need(option):
        walk_thresh = dict_settings['walk_thresh'][option]
        if walk_dist <= walk_thresh and option != 'sidewalk':
            err_message = 'this o-d pair is walkable (' + str(round(walk_dist, 0)) + ') mile, no need to take transit'
            print('Trip', row['trip_id'], err_message)
            return pd.DataFrame(
                {'trip_id': [row['trip_id']], 'option': [option], 'state': [err_message], 'numRoutes': [0],
                 'runTime': [time.time() - t1]})
        else:
            return pd.DataFrame()

    # When debugging, could comment out try/exception structure to better track the error.
    # try:
    t1 = time.time()
    walk_dist = np.sqrt((row['ox'] - row['dx']) ** 2 + (row['oy'] - row['dy']) ** 2) / 5280.0

    resultsPath = pd.DataFrame()
    runningLog = pd.DataFrame()

    # There is only one case for BikewaySim: walk at both ends of the trip from OD point to
    # nearest node on the network, and only biking on the network (O==>walk==>bike==>walk==>D)
    for option in options:
        # df: trip_id, option, state, numRoutes, runTime
        # if not walkable, return empty df
        checkWalk = determine_need(option)
        if len(checkWalk) == 0:
            resultsPathi, runningLogi = Bike_route_finder(row, option, dict_settings)
            resultsPath = pd.concat([resultsPath, resultsPathi], ignore_index=True)
            runningLog = pd.concat([runningLog, runningLogi], ignore_index=True)
        else:
            runningLog = pd.concat([runningLog,checkWalk], ignore_index=True)

        resultsPath['trip_id'] = row['trip_id']

    # block try except and unindent the code above when debugging
    # except Exception as e:
    #     err_message = ' has code problem'
    #     print(err_message, ':', e)  # just print out the error message
    #     print('Trip', row['trip_id'], err_message)
    #     runningLog = runningLog.append(pd.DataFrame(
    #         {'trip_id': [row['trip_id']], 'state': [err_message], 'numRoutes': [0], 'runTime': [time.time() - t1]}),
    #         ignore_index=True)

    return resultsPath, runningLog


def plot_seq_abm(trip_seq, gpd_df, skeleton=None, xlim=None, ylim=None):
    print(xlim, ylim)
    """
    trip_seq: a list of link id (includes 'origin' and 'destination')
    gdf_df: Geopandas dataframe (shapefile) of abm15
    skeleton: a list of link ID. if not None, those will be plotted as skeleton traffic network (e.g., I-85).
        If None,
    """
    col = 'NAME'

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    # convert sequential node ID to link ID
    As, Bs = trip_seq[0:-2:1], trip_seq[1:-1:1]
    trip_seq = [min(a,b) + '_' + max(a,b) for a, b in zip(As, Bs)]  # link IDs

    print('seq')
    print('len of seq is:', len(trip_seq))
    print(trip_seq)
    # trip_seq = trip_seq[1:-2:1]

    trip_df = gpd_df.loc[gpd_df['A_B'].isin(trip_seq), ]
    print('num of links found is:', len(trip_df))
    # for now, no skeleton for sidewalksim
    if skeleton is None:
        trip_df_skeleton = gpd_df.loc[gpd_df[col].str.contains('I-', na=False), ]
    else:  # input should be a list of names
        trip_df_skeleton = gpd_df.loc[gpd_df[col].isin(skeleton), ]
    trip_df_skeleton.plot(ax=ax, color='gray', alpha=0.2)

    # use merge method to keep the sequence of trajectory correct
    trip_seq = pd.DataFrame(trip_seq, columns=['seq'])
    trip_df = trip_df.merge(trip_seq, left_on='A_B', right_on='seq', how='left')
    trip_df.plot(ax=ax, color='red', alpha=0.4)  # (column='SPEEDLIMIT')
    # Set limints, if given
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    # create legend manually
    if skeleton is not None:
        skeleton_gray = mpatches.Patch(color='gray', label='skeleton network')
    trip_legend_red = mpatches.Patch(color='red', label='trip trajectory')
    if skeleton is not None:
        skeleton_gray = mpatches.Patch(color='gray', label='skeleton network')
        plt.legend(handles=[skeleton_gray, trip_legend_red])
    else:
        plt.legend(handles=[trip_legend_red])

    return plt


def plot_seq_sw(trip_seq, gpd_df, trip_series=None, xlim=None, ylim=None):
    """
    trip_seq: a list of link id (includes 'origin' and 'destination')
    gdf_df: Geopandas dataframe (shapefile) of abm15
    skeleton: a list of link ID. if not None, those will be plotted as skeleton traffic network (e.g., I-85).
        If None,
    """
    # import parcel and plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    # convert sequential node ID to link ID
    As, Bs = trip_seq[0:-2:1], trip_seq[1:-1:1]
    trip_seq = [min(a,b) + '_' + max(a,b) for a, b in zip(As, Bs)]  # link IDs

    print('seq')
    print(trip_seq)
    print('len of seq is:', len(trip_seq))
    # trip_seq = trip_seq[1:-2:1]

    trip_df = gpd_df.loc[gpd_df['A_B'].isin(trip_seq), ]
    print('num of links found is:', len(trip_df))
    # for now, no skeleton for sidewalksim

    # use merge method to keep the sequence of trajectory correct
    trip_seq = pd.DataFrame(trip_seq, columns=['seq'])
    trip_df = trip_df.merge(trip_seq, left_on='A_B', right_on='seq', how='left')

    # plot origin/destination
    if trip_series is not None:
        t1 = trip_series
        ax.plot(t1['ox'], t1['oy'], 'ko', alpha=0.4, label='origin')  # origin
        ax.plot(t1['dx'], t1['dy'], 'k^', alpha=0.4, label='destination')  # destination

    trip_sw = trip_df.loc[trip_df['type'] == 'sidewalk', ]
    trip_cw = trip_df.loc[trip_df['type'] == 'crosswalk', ]

    
    # Set the x and y limits if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    gpd_df.plot(ax=ax, color='grey', alpha=0.4, label='backgroud')
    trip_sw.plot(ax=ax, color='green', alpha=1, linewidth=3, label='sidewalk')  # (column='SPEEDLIMIT')
    trip_cw.plot(ax=ax, color='red', alpha=1, linewidth=3, label='crosswalk')
    plt.legend()
    return plt


def allRun(df_points, options, dict_settings, rootpath, use_impedance=False, col=None):
    # return csv file names with datetime info
    from datetime import datetime
    now = datetime.now()  # used time to uniquely name outputs sub-folder
    now_str = now.strftime("%y%m%d_%H%M%S")
    new_folder = os.path.join(rootpath, 'build_graph/results/{}'.format(col))
    new_plot_folder = os.path.join(rootpath, 'build_graph/results_route/{}'.format(col))  # used to store folders
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    if not os.path.exists(new_plot_folder):
        os.makedirs(new_plot_folder)
    dict_settings['plot_folder'] = new_plot_folder

    # Use plot limit if specified
    if dict_settings['plot_limit'] == True:
        #
        max_x = max(df_points['ox'].max(), df_points['dx'].max())
        min_x = min(df_points['ox'].min(), df_points['dx'].min())
        max_y = max(df_points['oy'].max(), df_points['dy'].max())
        min_y = min(df_points['oy'].min(), df_points['dy'].min())
        #
        buffer_x = (max_x - min_x)/10
        buffer_y = (max_y - min_y)/10
        #
        max_x = max_x + buffer_x
        min_x = min_x - buffer_x
        max_y = max_y + buffer_y
        min_y = min_y - buffer_y
        #
        dict_settings['limits'] = {'xlim':(min_x, max_x), 'ylim':(min_y, max_y)}

    results, logs = pd.DataFrame(), pd.DataFrame()
    t = []
    trips = []
    for ind, row in df_points.iterrows():
        ti = time.time()
        resultsi, logsi = BIGRun(row, options, dict_settings)
        # results = results.append(resultsi) # Fizzy's Edit: November 2021
        logs = pd.concat([logs,logsi], ignore_index=True)
        t.append(time.time() - ti)
        trips.append(row['trip_id'])
        resultsi.to_csv(os.path.join(new_folder, 'paths_walk_{}.csv'.format(ind)), index=False)
    # 
    logs.to_csv(os.path.join(new_folder, 'logs_walk.csv'), index=False)
    log_sum = pd.DataFrame({'trip_id': trips, 'time': t})
    log_sum.to_csv(os.path.join(new_folder, 'log_sum_walk.csv'), index=False)


# Useless for now.
def errMessages(row, option, condition):
    """
    NOT USED for now. This function is used for different error information for TransitSim.
    TODO: Could change this function for different scenarios for the Bike case in the future.
    """
    if condition == 1:
        numRoutes, err_message = 0, 'no [' + option + '] stops could be found from destination that are within ' \
                                                      'walking distance '
        print('Trip', row['trip_id'], err_message)
    if condition == 2:
        numRoutes, err_message = 0, 'no [' + option + '] stops could be found from origin that are within walking ' \
                                                      'distance & time '
        print('Trip', row['trip_id'], err_message)
    return numRoutes, err_message


def main_BikewaySim():
    os.environ['PROJ_LIB'] = '/Volumes/Samsung_T5/Courses/CEE6701_planning/Scripts/02_SidewalkSim'
    dirPath = os.path.join(os.environ['PROJ_LIB'], 'build_graph')
    os.chdir(dirPath)

    # Section 1: BUILD GRAPHS & ASSUMPTIONS
    print('** Initialize bike network **')
    try:
        df_links = gpd.read_file(os.path.join('data_node_link', 'abm15_links.shp'))
    except:
        df_links = initialize_abm15_links()

    print('** build bikeway network **')
    dict_bike = {'DG': build_bike_network(df_links), 'links': df_links}
    # combine settings
    dict_settings = {}
    # strategy: 1. given origin time find earliest arrival
    # 2. given expected arrival time find latest departure time
    dict_settings = {'walk_speed': 2.0,  # people's walking speed 2 mph
                     'grid_size': 25000.0,  # search links using grids with grid width 25000 feet
                     'ntp_dist_thresh': 5280.0,  # node to point (origin/destination) distance threshold
                     'network': {'bike': dict_bike},  # dump in networks and modes
                     'strategy': {'bike': 1},  # 1. find earliest arrival 2. find latest departure
                     'query_time': [8],  # departure time or arrival time of a trip
                     'walk_thresh': {'bike': 1},  # walking threshold is 1 mile
                     'num_options': {'bike': 2},  # can return k-shortest paths instead of one
                     'plot_all': True,  # plot and save plots for all routes found
                     'one_by_one': False  # set time and strategy one by one
                     }
    print('** load trip data & prepare sample **')
    sample_Prepared = True
    options = ['bike']

    df_points = pd.DataFrame()

    # Section 2: lOAD SAMPLE DATA; could have multiple csv sample files.
    if dict_settings['one_by_one'] is False:
        if sample_Prepared:
            for samp_file in glob.glob(os.path.join('samples_out', '*.csv')):
                dfi = pd.read_csv(samp_file)
                dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                df_points = pd.concat([df_points, dfi], ignore_index=True)
        else:
            for samp_file in glob.glob(os.path.join('samples_in', '*.csv')):
                df_points = pd.concat([df_points, samp_pre_process(samp_file, dict_settings)], ignore_index=True)
    else:
        if sample_Prepared:
            for samp_file in glob.glob(os.path.join('one_by_one_out', '*.csv')):
                dfi = pd.read_csv(samp_file)
                dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                df_points = pd.concat([df_points, dfi], ignore_index=True)
        else:
            for samp_file in glob.glob(os.path.join('one_by_one_in', '*.csv')):
                df_points = pd.concat([df_points, samp_pre_process(samp_file, dict_settings)], ignore_index=True)

    allRun(df_points, options, dict_settings)


def main_SidewalkSim():
    os.environ['PROJ_LIB'] = '/Volumes/Samsung_T5/Courses/CEE6701_planning/Scripts/02_SidewalkSim'
    dirPath = os.path.join(os.environ['PROJ_LIB'], 'build_graph')
    os.chdir(dirPath)

    # Section 1: BUILD GRAPHS & ASSUMPTIONS
    print('** Initialize bike network **')
    # try:
    #     sw_links = gpd.read_file(os.path.join(os.environ['PROJ_LIB'],
    #                                           'build_graph', 'data_node_link', 'SWS_sw_links.shp'))
    #     cw_links = gpd.read_file(os.path.join(os.environ['PROJ_LIB'],
    #                                           'build_graph', 'data_node_link', 'SWS_cw_links.shp'))
    # except:
    df_links = initialize_sws_links()
    print('** build walking network **')
    dict_walk = {'DG': build_walk_network(df_links), 'links': df_links}

    # combine all settings
    dict_settings = {}

    # strategy: 1. given origin time find earliest arrival
    #           2. given expected arrival time find latest departure time
    dict_settings = {'walk_speed': 2.5,  # people's walking speed 2 mph
                     'grid_size': 25000.0,
                     # for searching nearby links by grouping links to grids with width 25000 ft. for efficiency in searching
                     'ntp_dist_thresh': 5280,
                     # node to point (maximum distance access to network from origin/destination);
                     'network': {'sidewalk': dict_walk},  # dump in networks and modes
                     # strategy determines network link's direction.
                     # Strategy 1: Find earliest arrival given query time as departure time
                     # Strategy 2: Find latest departure time given query time as arrival time
                     'strategy': {'sidewalk': 1},  # 1. find earliest arrival 2. find latest departure
                     'query_time': [8],  # departure time or arrival time of a trip, depends on the strategy

                     'walk_thresh': {'sidewalk': 1},  # walking threshold is 1 mile (IGNORED by the Sidewalk option)
                     'num_options': {'sidewalk': 1},  # if set to 2, return 2-shortest paths
                     'plot_all': True,  # if True, plot results and save plots for all routes found
                     'one_by_one': False  # set time and strategy one by one
                     }
    print('** load trip data & prepare sample **')
    sample_Prepared = False  # check prepared process
    options = ['sidewalk']

    df_points = pd.DataFrame()
    # Section 2: lOAD SAMPLE DATA; could have multiple csv sample files.
    if dict_settings['one_by_one'] is False:
        if sample_Prepared:
            for samp_file in glob.glob(os.path.join('samples_out', '*.csv')):
                dfi = pd.read_csv(samp_file)
                dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                df_points = pd.concat([df_points, dfi], ignore_index=True)
        else:
            for samp_file in glob.glob(os.path.join('samples_in', '*.csv')):
                df_points = pd.concat([df_points, samp_pre_process(samp_file, dict_settings, option='sidewalk')],
                                             ignore_index=True)
    else:
        if sample_Prepared:
            for samp_file in glob.glob(os.path.join('one_by_one_out', '*.csv')):
                dfi = pd.read_csv(samp_file)
                dfi['sample'] = os.path.basename(samp_file).split('.csv')[0]
                df_points = pd.concat([df_points, dfi], ignore_index=True)
        else:
            for samp_file in glob.glob(os.path.join('one_by_one_in', '*.csv')):
                df_points = pd.concat([df_points, samp_pre_process(samp_file, dict_settings, option='sidewalk')],
                                             ignore_index=True)
    allRun(df_points, options, dict_settings)


if __name__ == '__main__':
    # I prefer to use Jupyter notebook running it in several blocks,
    # instead of running the file at once.
    # The notebook can be set to auto-load this file/module every time
    # you make changes in this module and saved it.
    # But one can also just run the file directly through Terminal/Command Lines.

    # remember to reset the project base directory "os.environ['PROJ_LIB']" when needed.
    # os.environ['PROJ_LIB'] = '/Volumes/Samsung_T5/Courses/CEE6701_planning/Scripts/02_SidewalkSim'
    # main_BikewaySim()
    main_SidewalkSim()

