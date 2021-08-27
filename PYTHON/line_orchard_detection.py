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


def is_orchard_line(points):
    points.reset_index(drop=True, inplace=True)
    points.set_index(points.index + 1, inplace=True)
    points['id'] = points.index

    n_array = np.array(list(points.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(n_array)
    n_nearest = 15
    dist, idx = btree.query(n_array, k=n_nearest)

    for i in range(1, n_nearest):
        points['pt_{}'.format(i)] = idx[:, i] + 1

    saida_linhas = []
    for i in points.index:

        first = points.loc[points.loc[i, 'pt_1'], 'geometry']
        second = points.loc[points.loc[i, 'pt_2'], 'geometry']

        if is_line(first, points.loc[i, 'geometry'], second):
            saida_linhas.append(LineString([first, points.loc[i, 'geometry'], second]))
    #             points.at[i, 'middle'] = True

    return saida_linhas, points

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
    return df_buffer


def buffer_to_lines(buffer, points, linhas=None):
    if linhas is None:
        linhas = []

    points['buffer'] = -99

    for i in buffer.index:

        internos = points.geometry.within(buffer.loc[i, 'geometry'])
        select = points.loc[list([internos[internos].index][0])]

        x = np.array(list(select['geometry'].x))
        y = np.array(list(select['geometry'].y))

        points_selected = np.array([[x, y] for x, y in zip(list(select['geometry'].x), list(select['geometry'].y))])

        x_sorted = x.copy()
        x_sorted.sort()

        xy = []
        for z in x_sorted:
            xy.append(list(points_selected[points_selected[:, 0] == z][0]))

        points.at[list([internos[internos].index][0]), 'buffer'] = int(i)

        linhas.append(LineString(xy))

    gdf_linhas = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(linhas, crs=points.crs)})
    gdf_linhas['id'] = buffer.index
    gdf_linhas.set_index('id', inplace=True)
    return gdf_linhas


def mapping_lines(points, lines):
    #     points['line'] = -99
    points['line'] = ''

    for i in lines.index:

        # points_selected = points.loc[points['buffer'] == i]
        line = lines.loc[i, 'geometry']

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


def select_bridge(possiveis_pontes):
    """
    Encontra a ponte mais adequada, segundo as condições definidas, que liga o extremo da linha avaliada ao proximo
    ponto. O ponto a ser ligado pela ponte escolhida deve ser um extremo de outra linha, ou um ponto desprendido até
    então.

    :param possiveis_pontes: lista de listas dos tres pontos formadores das pontes ([Point0, Point1, Point2] [,...])
    :return:
    """
    mapeamento = {}
    for ids, i in enumerate(possiveis_pontes):
        # Para cada linha, são computadas as duas métricas abaixo:
        # Fracao: Razao entre o segundo e o primeiro segmento formador da ponte
        fracao = LineString([i[1], i[2]]).length / LineString([i[0], i[1]]).length
        # Angle: Angulo formado entre os tres pontos formadores da ponte
        angle = get_angle_vertex(i[0], i[1], i[2])

        mapeamento[ids] = {'fracao': fracao, 'angle': angle}

    # Dataframe com as metricas computadas. O indice do df segue a indexação da lista de entrada da funcao
    df_select = pd.DataFrame(mapeamento).T
    df_select.set_index(pd.Index([x for x in range(df_select.shape[0])]), inplace=True)

    ## Filtragem inicial ##
    # Exclusão das pontes em que a fracao é maior que 5
    '''Isso significa que só serao consideradas as pontes que ligam o extremo das linhas 
    com os pontos que estao até 5x a distancia entre os seus dois ultimos pontos do extremo analisado'''
    id_drop = df_select.loc[df_select['fracao'] > 5].index
    df_select = df_select.drop(index=id_drop, axis=0)
    # Fim da filtragem inicial ##

    # Ordenação do dataframe pela metrica fracao
    df_select = df_select.sort_values('fracao')

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
        segundo segmento muito proximos (30%), o fator de escolha se torna o angulo fodo pela ponte. Aquela de menor
        angulaçao entre os seus tres pontos e a escolhida'''

        menor_1 = df_select.iloc[0]['fracao']
        menor_2 = df_select.iloc[1]['fracao']

        if (menor_2 / menor_1) <= 1.3:
            index = df_select.head(2).sort_values('angle').iloc[[0]].index[0]
        else:
            index = df_select.head(1).index[0]
    ## Fim da selecao da ponte ##

    bridge = possiveis_pontes[index]

    return bridge


def snap_lines(points, lines):
    linhas_ponte = []
    for i in lines.index:

        selected = points.loc[points['buffer'] == i]

        for z in ['first', 'last']:

            point_1 = selected.loc[selected['line'] == z]

            if selected.shape[0] > 3:
                point_0_label = z + '_nearest'
            else:
                point_0_label = 'nearest'

            point_0 = selected.loc[selected['line'] == point_0_label]

            possiveis_pontes = []
            for y in range(1, 15):

                num_point = point_1.iloc[0]['pt_{}'.format(y)]

                point_2 = points.loc[points['id'] == num_point]

                angle = check_angle_from_distance(point_0.iloc[0].geometry,
                                                  point_1.iloc[0].geometry,
                                                  point_2.iloc[0].geometry)

                if is_line(point_0.iloc[0].geometry, point_1.iloc[0].geometry, point_2.iloc[0].geometry, angle):

                    # Ponto encontrado corresponde a um extremo
                    if point_2['line'].iloc[0] in ['first', 'last']:

                        stop = False
                        pt_sufix = 1
                        while not stop:
                            point_aux = points.loc[points['id'] == point_2['pt_{}'.format(pt_sufix)].iloc[0]]

                            if 'nearest' in point_aux['line'].iloc[0]:
                                stop = True
                            pt_sufix += 1

                        point_aux_geom = point_aux['geometry'].iloc[0]

                        if is_line(point_1.iloc[0].geometry, point_2.iloc[0].geometry, point_aux_geom, 30):
                            possiveis_pontes.append([point_0.iloc[0].geometry,
                                                     point_1.iloc[0].geometry,
                                                     point_2.iloc[0].geometry])

                    # Ponto encontrado esta solto
                    elif point_2['buffer'].iloc[0] == -99:
                        possiveis_pontes.append([point_0.iloc[0].geometry,
                                                 point_1.iloc[0].geometry,
                                                 point_2.iloc[0].geometry])

            if len(possiveis_pontes) <= 0:
                pass
            else:
                linha_escolhida = select_bridge(possiveis_pontes)

                if linha_escolhida is not None:
                    linhas_ponte.append(LineString(linha_escolhida))
    return linhas_ponte


def falhas_detect(linhas):
    falhas = gpd.GeoSeries()
    for i in linhas['geometry']:

        seg = list(map(LineString, zip(i.coords[:-1], i.coords[1:])))
        segments = gpd.GeoSeries(seg, crs=linhas.crs)

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
                    falha = maiores.loc[[idx]].interpolate(y, normalized=True)
                    falhas = pd.concat([falhas, falha], axis=0)
    return falhas


def detect_lines_uni(df_points):

    # df_points = get_centroids(canopy)

    linhas, points = is_orchard_line(df_points)
    geodf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(linhas, crs=df_points.crs)})
    # geodf = extrapolar_linhas(geodf)
    buffer = get_buffer(geodf)
    lines = buffer_to_lines(buffer, points)
    points = mapping_lines(points, lines)

    points_new = gpd.GeoDataFrame()

    stop = True
    while stop:

        if not points_new.empty:
            points = points_new.copy()

        linhas_novas = snap_lines(points, lines)

        if len(linhas_novas) > 0:
            #             print(len(linhas))
            #             print(len(linhas_novas))
            for i in linhas_novas:
                linhas.append(i)

            geodf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(linhas, crs=points.crs)})
            # geodf = extrapolar_linhas(geodf)
            buffer = get_buffer(geodf)
            lines = buffer_to_lines(buffer, points)
            points_new = mapping_lines(points, lines)
        else:
            stop = False

    falhas = falhas_detect(lines)
    return falhas, lines


def detect_lines(df_points, df_talhoes):

    todas_falhas = []
    todas_linhas = []

    for i in df_talhoes.index:

        points_selected = df_points.within(df_talhoes.loc[i, 'geometry'])
        df_points_selected = df_points.loc[points_selected]

        falhas, linhas = detect_lines_uni(df_points_selected)

        todas_linhas.append(linhas)
        todas_falhas.append(falhas)

    return todas_falhas, todas_linhas


def detect_lines_path(path):

    pontos = gpd.read_file(path + 'centroids.geojson')
    talhoes = gpd.read_file(path + 'talhoes.geojson')
    talhoes.set_index(talhoes.index+1, inplace=True)

    falhas_completas, linhas_completas = detect_lines(pontos, talhoes)

    for i, (falhas, linhas) in enumerate(zip(falhas_completas, linhas_completas)):
        falhas.to_file(path + 'gaps.gpkg', layer='{}'.format(i), driver='GPKG')
        linhas.to_file(path + 'lines.gpkg', layer='{}'.format(i), driver='GPKG')
