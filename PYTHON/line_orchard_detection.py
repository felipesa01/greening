import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

pd.options.mode.chained_assignment = None  # default='warn'


def get_angle(p1, p2):
    y = p2.y - p1.y
    x = p2.x - p1.x

    angle = np.degrees(np.arctan2(y, x))

    return angle


def get_angle_vertex(p1, p_middle, p2):
    get_angle(p_middle, p1)
    get_angle(p_middle, p2)

    angle = get_angle(p_middle, p1) - get_angle(p_middle, p2)

    if angle < 0:
        angle += 360

    return angle


def is_line(p1, p_middle, p2, angle_threshold=15):
    angle = get_angle_vertex(p1, p_middle, p2)

    if abs(angle - 180) <= angle_threshold:
        return True
    else:
        return False


def get_extrapoled_line(p1, p2):
    """Creates a line extrapoled in p1->p2 direction"""
    extrapol_ratio = 1
    # a = p1
    b = (p1[0] + extrapol_ratio * (p2[0] - p1[0]), p1[1] + extrapol_ratio * (p2[1] - p1[1]))
    return b


def round_school(x):
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def get_centroids(canopy):
    return gpd.GeoDataFrame({'geometry': canopy.centroid}, crs=canopy.crs)


def row_segments(points):
    points.reset_index(drop=True, inplace=True)
    points.set_index(points.index + 1, inplace=True)
    points['id'] = points.index

    n_array = np.array(list(points.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_array)
    n_nearest = 16
    dist, idx = btree.query(n_array, k=n_nearest)

    for i in range(1, n_nearest):
        points['pt_{}'.format(i)] = idx[:, i] + 1

    segments = []
    for i in points.index:

        first = points.loc[points.loc[i, 'pt_1'], 'geometry']
        second = points.loc[points.loc[i, 'pt_2'], 'geometry']

        if is_line(first, points.loc[i, 'geometry'], second):
            segments.append(LineString([first, points.loc[i, 'geometry'], second]))
    #             points.at[i, 'middle'] = True

    return segments, points


# Nao mais utilizada
def extrapolar_linhas(gdf):
    for i in gdf.index:
        coords = [[x, y] for x, y in zip(gdf.loc[i, 'geometry'].xy[0], gdf.loc[i, 'geometry'].xy[1])]
        coords.append(['first'])
        coords.append(['last'])
        coords[0], coords[1], coords[2], coords[3] = coords[-2], coords[0], coords[1], coords[2]

        coords[0] = get_extrapoled_line(coords[2], coords[1])
        coords[-1] = get_extrapoled_line(coords[2], coords[3])

        gdf.at[i, 'geometry'] = LineString(coords)
    return gdf


def get_buffer(geodf):
    buffer = geodf['geometry'].buffer(0.5).unary_union  # .explode().reset_index(drop=True)
    df_buffer = gpd.GeoDataFrame({"geometry": buffer}, crs=geodf.crs)
    df_buffer.reset_index(drop=True)
    # df_buffer.index = df_buffer.index + 1
    return df_buffer


def sort_rows(rows):
    rows['id_sort'] = -99

    x = np.array([x.coords.xy[0][int(len(x.coords.xy[0]) / 2)] for x in rows.geometry.values])
    y = np.array([x.coords.xy[1][int(len(x.coords.xy[1]) / 2)] for x in rows.geometry.values])

    rows['1st_coord'] = [str([round(x, 5), round(y, 5)]) for x, y in zip(x, y)]

    points_selected = np.array([[x.coords.xy[0][int(len(x.coords.xy[0]) / 2)],
                                 x.coords.xy[1][int(len(x.coords.xy[1]) / 2)]] for x in rows.geometry.values])

    x_sorted = x.copy()
    x_sorted.sort()

    y_sorted = y.copy()
    y_sorted.sort()

    # Computar amplitudes
    delta_x = x_sorted[-1] - x_sorted[0]
    delta_y = y_sorted[-1] - y_sorted[0]

    xy = []
    if delta_x >= delta_y:
        for z in x_sorted:
            # gdf_linhas.at[gdf_linhas.geometry.values[0].coords.xy[0] == z]
            xy.append(list(points_selected[points_selected[:, 0] == z][0]))
    else:
        for z in y_sorted:
            # gdf_linhas.at[gdf_linhas.geometry.values[0].coords.xy[1] == z]
            xy.append(list(points_selected[points_selected[:, 1] == z][0]))

    xy = [str([round(i[0], 5), round(i[1], 5)]) for i in xy]

    for i, coord in enumerate(xy):
        rows.at[rows['1st_coord'] == coord, 'id_sort'] = i + 1

    rows.drop('1st_coord', axis=1, inplace=True)

    return rows


def buffer_to_rows(buffer, points, pt_inverse=False):
    rows = []
    points['buffer'] = -99
    points['id_row'] = -99

    # id_row = list(range(buffer.shape[0]))
    # id_row_inverse = id_row[::-1]
    # both = [[a, b] for a, b in zip(id_row, id_row_inverse)]
    # id_row_mapping = {}
    # for i in both:
    #     if pt_inverse:
    #         id_row_mapping[i[0]] = i[1]
    #     else:
    #         id_row_mapping[i[0]] = i[0]

    for i in buffer.index:

        points_within = points.geometry.within(buffer.loc[i, 'geometry'])
        df_points = points.loc[[points_within[points_within].index][0]]

        x = np.array(df_points['geometry'].x)
        y = np.array(df_points['geometry'].y)

        xy_points = np.array([[x, y] for x, y in zip(df_points['geometry'].x, df_points['geometry'].y)])

        x_sorted = x.copy()
        x_sorted.sort()
        y_sorted = y.copy()
        y_sorted.sort()

        # Computar amplitudes
        delta_x = x_sorted[-1] - x_sorted[0]
        delta_y = y_sorted[-1] - y_sorted[0]

        xy = []
        if delta_x >= delta_y:
            for z in x_sorted:
                xy.append(list(xy_points[xy_points[:, 0] == z][0]))
        else:
            for z in y_sorted:
                xy.append(list(xy_points[xy_points[:, 1] == z][0]))

        id_points_in_row = [i + 1 for i in range(points_within[points_within].shape[0])]
        if pt_inverse:
            id_points_in_row.reverse()

        points.at[list([points_within[points_within].index][0]), 'id_row'] = id_points_in_row
        points.at[list([points_within[points_within].index][0]), 'buffer'] = int(i)

        rows.append(LineString(xy))

    df_rows = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(rows, crs=points.crs)})
    # df_rows.reset_index(drop=True, inplace=True)
    df_rows = sort_rows(df_rows)

    return df_rows


def mapping_rows(points, rows):
    #     points['line'] = -99
    points['line'] = ''

    for i in rows.index:

        # points_selected = points.loc[points['buffer'] == i]
        line = rows.loc[i, 'geometry']

        xy_line = [[x, y] for x, y in zip(line.xy[0], line.xy[1])]

        first_point = Point(xy_line[0])
        nea_first_point = Point(xy_line[1])

        last_point = Point(xy_line[-1])
        nea_last_point = Point(xy_line[-2])

        id_first = points.loc[points['geometry'] == first_point].index[0]
        id_nea_first = points.loc[points['geometry'] == nea_first_point].index[0]

        id_last = points.loc[points['geometry'] == last_point].index[0]
        id_nea_last = points.loc[points['geometry'] == nea_last_point].index[0]

        points.at[id_first, 'line'] = 'first'
        points.at[id_last, 'line'] = 'last'

        if id_nea_first == id_nea_last:
            points.at[id_nea_first, 'line'] = 'nearest'
        else:
            points.at[id_nea_first, 'line'] = 'first_nearest'
            points.at[id_nea_last, 'line'] = 'last_nearest'

    return points


def check_angle_from_distance(p0, p1, p2):
    segment_origin = LineString([p0, p1])
    segment_target = LineString([p1, p2])

    ratio = segment_target.length / segment_origin.length

    # Encontrei a eq. da reta aproximada formada pelas condições abaixo
    # if ratio <= 0.7:
    #     angle = 90
    # elif ratio <= 1:
    #     angle = 45
    # elif ratio <= 2:
    #     angle = 30
    # elif ratio <= 3:
    #     angle = 20
    # elif ratio <= 4:
    #     angle = 15
    # else:
    #     angle = 10

    if ratio < 0.7:
        angle = 90
    elif ratio > 4:
        angle = 15
    else:
        angle = 52.88450752 * ratio ** -0.8643306

    return angle


def select_bridge(potential_bridges):
    """
    Encontra a ponte mais adequada, segundo as condições definidas, que liga o extremo da linha avaliada ao proximo
    ponto. O ponto a ser ligado pela ponte escolhida deve ser um extremo de outra linha ou um ponto isolado até
    então.

    :param potential_bridges: lista de listas dos tres pontos formadores das pontes ([Point0, Point1, Point2] [,...])
    :return:
    """
    metrics = {}
    for ids, i in enumerate(potential_bridges):
        # Para cada linha, são computadas as duas métricas abaixo:
        # Ratio: Razao entre o segundo e o primeiro segmento formador da ponte
        ratio = LineString([i[1], i[2]]).length / LineString([i[0], i[1]]).length
        # Angle: Angulo formado entre os tres pontos formadores da ponte
        angle = get_angle_vertex(i[0], i[1], i[2])

        metrics[ids] = {'ratio': ratio, 'angle': angle}

    # Dataframe com as metricas computadas. O indice do df segue a indexação da lista de entrada da funcao
    df_select = pd.DataFrame(metrics).T
    df_select.set_index(pd.Index([x for x in range(df_select.shape[0])]), inplace=True)

    ## Filtragem inicial ##
    # Exclusão das pontes em que o ratio é maior que 5
    '''Isso significa que só serao consideradas as pontes que ligam o extremo das linhas 
    com os pontos que estao até 5x a distancia entre os seus dois ultimos pontos do extremo analisado'''
    id_drop = df_select.loc[df_select['ratio'] > 5].index
    df_select = df_select.drop(index=id_drop, axis=0)
    # Fim da filtragem inicial ##

    # Ordenação do dataframe pela metrica ratio
    df_select = df_select.sort_values('ratio')

    # Caso não haja remanescentes da filtragem, retona-se None
    if df_select.empty:
        return None
    # Caso reste apenas uma ponte, ela é a selecionada
    elif df_select.shape[0] == 1:
        index = df_select.iloc[[0]].index[0]  # Aquisição inadequada do valor da célula (MELHORAR!!)
    ## Selecão da ponte entre as remanescentes ##
    else:
        # Regras de selecao #
        '''O fator prioritário na seleção da ponte é a distância do segundo segmento. Quando mais próximo do primeiro,
        ou quanto menor a fracão, mais adequado. Entretanto, nos casos em que duas pontes apresentem tamanhos do
        segundo segmento muito proximos (30%), o fator de escolha se torna o angulo formado pela ponte. Aquela de menor
        angulaçao entre os seus tres pontos e a escolhida'''

        menor_1 = df_select.iloc[0]['ratio']
        menor_2 = df_select.iloc[1]['ratio']

        if (menor_2 / menor_1) <= 1.3:
            index = df_select.head(2).sort_values('angle').iloc[[0]].index[0]
        else:
            index = df_select.head(1).index[0]
    ## Fim da selecao da ponte ##

    bridge = potential_bridges[index]

    return bridge


def snap_rows(points, rows):
    bridges = []
    for i in rows.index:

        selected = points.loc[points['buffer'] == i]

        # Busca em cada extremo
        for z in ['first', 'last']:

            point_1 = selected.loc[selected['line'] == z]

            if selected.shape[0] > 3:
                point_0_label = z + '_nearest'
            else:
                point_0_label = 'nearest'

            point_0 = selected.loc[selected['line'] == point_0_label]

            potential_bridges = []
            for y in range(1, 16):

                num_point = point_1.loc[point_1.index[0], 'pt_{}'.format(y)]
                point_2 = points.loc[points['id'] == num_point]

                angle = check_angle_from_distance(point_0.iloc[0].geometry,
                                                  point_1.iloc[0].geometry,
                                                  point_2.iloc[0].geometry)

                if is_line(point_0.iloc[0].geometry, point_1.iloc[0].geometry, point_2.iloc[0].geometry, angle):

                    # Ponto encontrado corresponde a um extremo
                    if point_2['line'].iloc[0] in ['first', 'last']:

                        stop = False
                        pt_sufix = 1
                        point_aux = 0  # Só pela integridade do codigo
                        while not stop:
                            point_aux = points.loc[points['id'] == point_2['pt_{}'.format(pt_sufix)].iloc[0]]

                            if 'nearest' in point_aux['line'].iloc[0]:
                                stop = True
                            pt_sufix += 1

                        point_aux_geom = point_aux['geometry'].iloc[0]

                        if is_line(point_1.iloc[0].geometry, point_2.iloc[0].geometry, point_aux_geom, 30):
                            potential_bridges.append([point_0.iloc[0].geometry,
                                                      point_1.iloc[0].geometry,
                                                      point_2.iloc[0].geometry])

                    # Ponto encontrado esta solto
                    elif point_2['buffer'].iloc[0] == -99:
                        potential_bridges.append([point_0.iloc[0].geometry,
                                                  point_1.iloc[0].geometry,
                                                  point_2.iloc[0].geometry])

            if len(potential_bridges) <= 0:
                pass
            else:
                selected_bridge = select_bridge(potential_bridges)

                if selected_bridge is not None:
                    bridges.append(LineString(selected_bridge))
    return bridges


def detect_gaps(points, rows):
    gaps = gpd.GeoDataFrame({'id': -99,
                             'buffer': -99,
                             'gap': -99,
                             'geometry': gpd.GeoSeries(crs=points.crs)})
    id_pt = list(points['id'])[-1] + 1
    for ids, i in zip(list(rows.index), rows['geometry']):

        seg = list(map(LineString, zip(i.coords[:-1], i.coords[1:])))
        segments = gpd.GeoSeries(seg, crs=rows.crs)

        dist = (segments.length.quantile(0.5) + segments.length.quantile(0.6)) / 2

        maiores = segments[segments.length > 1.05 * dist]

        if maiores.shape[0] == 0:
            continue
        else:
            target_lines = round((maiores.length / dist), 2)
            index = target_lines.index
            split = list(target_lines)

            for idx, z in zip(index, split):
                z = round_school(z)
                slices = pd.Series([x / z for x in range(1, z)])
                for y in slices:
                    gap = maiores.loc[[idx]].interpolate(y, normalized=True)
                    df_gap = gpd.GeoDataFrame({'id': id_pt,
                                               'buffer': ids,
                                               'gap': 1,
                                               'geometry': gap})
                    gaps = pd.concat([gaps, df_gap], axis=0)

                    id_pt += 1

    points_new = pd.concat([points, gaps], axis=0)

    return points_new


def detect_lines_uni(df_points, row_inverse=False, pt_inverse=False):
    # df_points = get_centroids(canopy)

    unit_segments, points = row_segments(df_points)
    df_segments = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(unit_segments, crs=df_points.crs)})
    buffer = get_buffer(df_segments)
    rows = buffer_to_rows(buffer, points)
    points = mapping_rows(points, rows)

    points_new = gpd.GeoDataFrame()

    stop = True
    while stop:

        if not points_new.empty:
            points = points_new.copy()

        new_segments = snap_rows(points, rows)

        if len(new_segments) > 0:
            for i in new_segments:
                unit_segments.append(i)

            df_segments = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(unit_segments, crs=points.crs)})
            buffer = get_buffer(df_segments)
            rows = buffer_to_rows(buffer, points)
            points_new = mapping_rows(points, rows)
        else:
            stop = False

    # Detectar falhas
    points_new['gap'] = 0
    points_new = detect_gaps(points_new, rows)
    rows = buffer_to_rows(buffer, points_new, pt_inverse=pt_inverse)

    ## Editar saidas ##
    # linhas
    rows.rename(columns={'id_sort': 'id'}, inplace=True)
    id_final = list(rows['id'])
    if row_inverse:
        # rows = rows.assign(id=rows.id.values[::-1])
        id_final.reverse()
    #
    rows['id'] = id_final

    # Centroides
    points_new.drop(['pt_{}'.format(i) for i in range(1, 16)], axis=1, inplace=True)
    points_new.drop('line', axis=1, inplace=True)
    points_new.rename(columns={'buffer': 'row'}, inplace=True)
    points_new.row = points_new.row + 1  # index 1-based

    if row_inverse:
        id_row = list(set(list(points_new.row.values)))
        id_row_inv = id_row[::-1]
        points_new.insert(4, 'row_ord', -99)
        for old, new in zip(id_row, id_row_inv):
            points_new.at[points_new['row'] == old, 'row_ord'] = new

        points_new.drop('row', axis=1, inplace=True)
        points_new.rename(columns={'row_ord': 'row'}, inplace=True)

    label_id = ['{row:03}-{pt:04}'.format(row=r, pt=p) for r, p in
                zip(list(points_new['row']), list(points_new['id_row']))]
    points_new.insert(1, 'label_id', label_id)

    return rows, points_new


# def detect_lines(df_points, df_talhoes):
#     all_gaps = []
#     all_rows = []
#
#     for i in df_talhoes.index:
#         points_selected = df_points.within(df_talhoes.loc[i, 'geometry'])
#         df_points_selected = df_points.loc[points_selected]
#
#         gaps, rows, points = detect_lines_uni(df_points_selected)
#
#         all_rows.append(rows)
#         all_gaps.append(gaps)
#
#     return all_gaps, all_rows, points
#
#
# def detect_lines_path(path):
#     pontos = gpd.read_file(path + 'centroids.geojson')
#     talhoes = gpd.read_file(path + 'talhoes.geojson')
#     talhoes.set_index(talhoes.index + 1, inplace=True)
#
#     falhas_completas, linhas_completas, points = detect_lines(pontos, talhoes)
#
#     points.to_file(path + 'centroids.geojson', driver='GeoJSON')
#     for i, (falhas, linhas) in enumerate(zip(falhas_completas, linhas_completas)):
#         falhas.to_file(path + 'gaps.gpkg', layer='{}'.format(i), driver='GPKG')
#         linhas.to_file(path + 'lines.gpkg', layer='{}'.format(i), driver='GPKG')
