import os
import sys
import rasterio
import rasterio.plot
from osgeo import gdal, osr, ogr
import shapely
from shapely import geometry
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from time import sleep
gdal.UseExceptions()


class PreProcessing:

    def __init__(self, ortomosaico_dir, copas_dir=None, proj_name=None):

        if proj_name is None:
            self.proj_name = os.path.splitext(os.path.basename(ortomosaico_dir))[0].lower()
        else:
            self.proj_name = proj_name

        self.proj_dir = os.path.join('../datasets', self.proj_name)
        if not os.path.isdir(self.proj_dir):
            os.mkdir(self.proj_dir)
            os.mkdir(os.path.join(self.proj_dir, 'img_patches'))

        self.orto = gdal.Open(ortomosaico_dir)  # Dataset da imagem
        self.grid = None
        if copas_dir is None:
            self.copas = None
        else:
            self.copas = gpd.read_file(copas_dir).set_index('id').sort_index()

    def split_img(self, patch_size=256, overlap=0, driver_grid='GeoJSON', save_img=True):
        # Acessar EPSG do mosaico
        prj = self.orto.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        epsg = 'EPSG:' + srs.GetAuthorityCode('projcs')

        # Start do ID e do dicionario de poligonos formadores da grade
        i = 0
        grid_pol = {'id': [], 'geometry': []}

        # Gerar iteradores do corte
        iter_x = range(0, self.orto.RasterXSize, patch_size - overlap)
        iter_y = range(0, self.orto.RasterYSize, patch_size - overlap)
        num_all_img = len(iter_x) * len(iter_y)

        with tqdm(total=num_all_img) as pbar:
            pbar.set_description("Cortando imagens")
            sleep(0.1)
            # print('Gerando patches')

            # Iterar em cada linha da imagem com passos de mesmo tamanho dos patches
            for px_x in iter_x:

                # Iterar em cada coluna da imagem com passos de mesmo tamanho dos patches
                for px_y in iter_y:

                    # Gerar o dataset do patch na memoria
                    ds = gdal.Translate('/vsimem/{i}.tif'.format(i=i),
                                        self.orto,
                                        srcWin=[px_x, px_y, patch_size, patch_size])

                    # Gerar poligono envolvente da imagem
                    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
                    width, height = ds.RasterXSize, ds.RasterYSize
                    xmax = xmin + width * xpixel
                    ymin = ymax + height * ypixel
                    geom = shapely.geometry.box(xmin, ymin, xmax, ymax)

                    # Caso seja declarado o arquivo de copas (para treinamento), cortar apenas os patches que contêm
                    # algum vetor no seu interior (condição do if)
                    if self.copas is not None:
                        # Avaliar se o patch corresponde a uma região da imagem com valores válidos (Não zero)
                        if len(ds.ReadAsArray().nonzero()[0]) != 0 and self.copas.intersects(geom).any():
                            if save_img:
                                # Salvar o pacth em disco
                                gdal.Translate(
                                    os.path.join(self.proj_dir, 'img_patches', '{i:05}.tif'.format(i=i)),
                                    self.orto,
                                    srcWin=[px_x, px_y, patch_size, patch_size])

                            # Inserir poligono na lista
                            grid_pol['id'].append(i)
                            grid_pol['geometry'].append(geom)

                            # Atualizar identificador
                            i += 1
                        pbar.update(1)
                    # Caso não seja declarado o arquivo de copas (apenas para inferência)
                    else:
                        # Avaliar se o patch corresponde a uma região da imagem com valores válidos (Não zero)
                        if len(ds.ReadAsArray().nonzero()[0]) != 0:
                            if save_img:
                                # Salvar o pacth em disco
                                gdal.Translate(
                                    os.path.join(self.proj_dir, 'img_patches', '{i:05}.tif'.format(i=i)),
                                    self.orto,
                                    srcWin=[px_x, px_y, patch_size, patch_size])

                            # Inserir poligono na lista
                            grid_pol['id'].append(i)
                            grid_pol['geometry'].append(geom)

                            # Atualizar identificador
                            i += 1
                        pbar.update(1)

        # Gerar o GeoDataFrame com a grade e salvá-la
        grid = gpd.GeoDataFrame(grid_pol, crs=epsg).set_index('id')
        grid.insert(0, 'split_samples', '')
        if self.copas is None:
            grid['split_samples'] = 'test'
        self.grid = grid
        grid.to_file(os.path.join(self.proj_dir, 'img_grid.geojson'), driver=driver_grid)

        print("Finished!")
        print("Total images: ", grid.shape[0])

    def coords_to_xy(self, corte, id_img):

        img = rasterio.open(os.path.join(self.proj_dir, 'img_patches', '{z:05}.tif'.format(z=id_img)))

        corte.insert(len(corte.columns) - 1, 'geometry_image', '')
        # corte['geometry_image'] = corte['geometry_image'].astype('string')

        for i in corte.index:

            # [FEITO] Preciso avaliar a ocorrencia de multipartes
            # A função .coords e .exterior não se aplicam a multipartes [FEITO]
            coord = list(corte.loc[i, 'geometry'].exterior.coords)

            segmentation = ''
            # Atenção aqui!! A visualização das annotations indicou que na lista de coors_img, o y precede o x
            for xy in coord:
                segmentation += str(round(img.index(xy[0], xy[1], float)[1], 2)) + ', ' + str(
                    round(img.index(xy[0], xy[1], float)[0], 2)) + ', '

            segmentation = segmentation[:-2]
            corte.at[i, 'geometry_image'] = segmentation

        return corte

    def is_geom_valid(self, corte, geom='MultiPolygon'):
        '''
        Criar uma grade com patches selecionados

        :param
        :return:
        '''

        features = []
        for i in corte.index:
            if corte.loc[i, 'geometry'].geom_type == geom:
                features.append(i)

        if len(features) != 0:
            return features
        else:
            return True

    def check_geom_split(self):
        list_empty = []
        dict_invalid = {}

        with tqdm(total=len(self.grid.index)) as pbar:
            pbar.set_description("Avaliando geometrias")
            sleep(0.1)
            for i in self.grid.index:
                corte = gpd.clip(self.copas, self.grid.loc[[i]])

                if corte.empty:
                    list_empty.append(i)
                else:
                    if type(self.is_geom_valid(corte)) == list:
                        dict_invalid['{i:05}'.format(i=i)] = self.is_geom_valid(corte)
                pbar.update(1)
        return dict_invalid

    def split_vector(self):

        dict_invalid = self.check_geom_split()

        if len(dict_invalid) == 0:
            with tqdm(total=len(self.grid.index)) as pbar:
                pbar.set_description("Salvando arquivos")
                sleep(0.1)
                for i in self.grid.index:
                    # Avaliar situação em que keep_geom (argumento de gpd.clip()) seja True
                    corte = gpd.clip(self.copas, self.grid.loc[[i]])
                    if not corte.empty:
                        corte = self.coords_to_xy(corte, self.grid.loc[i].name)
                        dir_out = os.path.join(self.proj_dir, 'vector_split.gpkg')
                        corte.to_file(dir_out, layer='{i:05}'.format(i=i), driver="GPKG")
                    pbar.update(1)
                print('Concluído!')
        else:
            print('Você deve editar os vetores dos seguintes patches para que não sejam gerados poligonos multipartes', dict_invalid)
            return

    def split_vector_2(self):

        with tqdm(total=len(self.grid.index)) as pbar:
            pbar.set_description("Salvando arquivos")
            sleep(0.1)
            for i in self.grid.index:
                # Avaliar situação em que keep_geom (argumento de gpd.clip()) seja True
                corte = gpd.clip(self.grid.loc[[i]], self.copas, keep_geom_type=True).explode().reset_index()
                if not corte.empty:
                    # corte = self.coords_to_xy(corte, self.grid.loc[i].name)
                    dir_out = os.path.join(self.proj_dir, 'vector_split.gpkg')
                    corte.to_file(dir_out, layer='{i:05}'.format(i=i), driver="GPKG")
                pbar.update(1)
            print('Concluído!')

    def is_all_same_geom(self, geom='MultiPolygon'):
        '''
        Criar uma grade com patches selecionados

        :param
        :return:
        '''

        # grid_selected = gpd.read_file(grade_dir).set_index('id')

        multi = {}
        for z in self.grid.index:

            corte = gpd.read_file(os.path.join(self.proj_dir, 'vector_split.gpkg'), layer='{z:05}'.format(z=z))
            features = []
            for i in range(corte.shape[0]):

                if corte.loc[i, 'geometry'].geom_type == geom:
                    features.append(i)

            if len(features) != 0:
                multi['{z:05}'.format(z=z)] = features

        if len(multi) == 0:
            return True
        else:
            return False, multi

    # def coords_to_xy_other(self, grid_selected=None):
    #
    #     if grid_selected == None:
    #         grid_selected = self.grid
    #
    #     for z in grid_selected.index:
    #
    #         img = rasterio.open('../IMGS/CORTES/{prefix}{z:05}.tif'.format(prefix=self.prefixo, z=z))
    #         corte = gpd.read_file('../VECTOR/CORTES/{prefix}{z:05}.geojson'.format(prefix=self.prefixo, z=z))
    #
    #         corte.insert(len(corte.columns) - 1, 'geometry_image', 0)
    #         corte['geometry_image'] = corte['geometry_image'].astype('string')
    #
    #         for i in corte.index:
    #
    #             # [FEITO] Preciso avaliar a ocorrencia de multipartes
    #             # A função .coords e .exterior não se aplicam a multipartes [FEITO]
    #             coord = list(corte.loc[i, 'geometry'].exterior.coords)
    #
    #             segmentation = ''
    #             # Atenção aqui!! A visualização das annotations indicou que na lista de coors_img, o y precede o x
    #             for xy in coord:
    #                 segmentation += str(round(img.index(xy[0], xy[1], float)[1], 2)) + ', ' + str(
    #                     round(img.index(xy[0], xy[1], float)[0], 2)) + ', '
    #
    #             segmentation = segmentation[:-2]
    #             corte.at[i, 'geometry_image'] = segmentation
    #
    #         corte.to_file('../VECTOR/CORTES/EDITADOS/corte_{}.geojson'.format(z), driver="GeoJSON")


class Processing:

    def readimagetif(self, image_path, str_array_type):
        """
        :param image_path:
        is the path for the image to be computed
        :param str_array_type:
        is the type of array Float, Integer, Byte
        :return: array, image_tif
        array is the nupy  array from image
        image_tif is the tif image read from disk
        """

        # read the geotiff image and return the numpy array
        # with the correct shape

        image_tif = gdal.Open(image_path)
        image_array = image_tif.ReadAsArray()
        bands = image_array.shape[0]
        rows = image_array.shape[1]
        cols = image_array.shape[2]

        if str_array_type == 'Float':
            array_type = np.float64
        elif str_array_type == 'Integer':
            array_type = np.uint16
        elif str_array_type == 'Byte':
            array_type = np.uint8
        else:
            print('Impossible run for this type')
            sys.exit()
        # end if

        array = np.empty([rows, cols, bands], dtype=array_type)
        for k in range(bands):
            array[:, :, k] = image_array[k, :, :]

        return array, image_tif

    def array2raster(self, raster_path, array_type, raster_origin, pixel_height,
                     pixel_width, rot_y, rot_x, drive, projection, array):
        """
        :param raster_path:
        :param raster_origin:
        :param pixel_height:
        :param pixel_width:
        :param rot_y:
        :param rot_x:
        :param drive:
        :param projection:
        :param array:
        :return:
        """
        rows = array.shape[0]
        cols = array.shape[1]

        if array.ndim == 2:
            bands = 1
        else:
            bands = array.shape[2]

        origin_y = raster_origin[0]
        origin_x = raster_origin[1]

        if array_type == 'Float':
            out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Float64)
        elif array_type == 'Byte':
            out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Byte)
        elif array_type == 'Integer':
            out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Int16)
        else:
            print('Type is incorrect')
            sys.exit()

        out_raster.SetGeoTransform((origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height))

        for i in range(bands):
            out_band = out_raster.GetRasterBand(i + 1)
            if bands == 1:
                out_band.WriteArray(array)
            else:
                out_band.WriteArray(array[:, :, i])
        # end for

        out_raster_srs = osr.SpatialReference(wkt=projection)
        out_raster.SetProjection(out_raster_srs.ExportToWkt())
        out_band.FlushCache()

        return 'Raster created!!!!'

    def raster2polygon(self, raster_path, shape_path, layer_name):
        """
        :param raster_path:
        :param shape_path:
        :param layer_name:
        :return:
        """

        src_ds = gdal.Open(raster_path)
        band = src_ds.GetRasterBand(1)
        mask = band
        driver = ogr.GetDriverByName('GeoJSON')
        dst_ds = driver.CreateDataSource(shape_path)
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(src_ds.GetProjectionRef())
        layer = dst_ds.CreateLayer(layer_name, srs=spatial_ref)
        fd = ogr.FieldDefn('ID', ogr.OFSTInt16)
        layer.CreateField(fd)
        dst_field = 0
        gdal.Polygonize(band, mask, layer, dst_field, [], None)

        return 'Shapefile created!!!!'


    def join_vectors(self, shape_path_aux, limiar=0.1):

        result_paths = []
        for filename in os.listdir(shape_path_aux):
            if os.path.splitext(filename)[1].lower() == '.geojson' and \
                    os.path.splitext(filename)[0].lower() != 'final_result':
                result_paths.append(os.path.join(shape_path_aux, filename))

        pols = pd.concat([gpd.read_file(i) for i in result_paths], axis=0).reset_index(drop=True)
        pols['ID'] = pols.index
        pols.insert(1, column='union', value=0)

        delete_lines = []
        for i in pols.index:
            if pols.loc[i, 'union'] == 0:
                polygon = pols.loc[i, 'geometry']
                intersec = pols.overlaps(polygon)
                intersec = intersec[intersec]
                list_result = list(intersec.index)

                if len(list_result) != 0:
                    maior = pols.loc[list_result, 'geometry'].area.idxmax(axis=0)

                    for z in list_result:
                        inter_poly = pols.loc[z, 'geometry'].intersection(pols.loc[maior, 'geometry'])
                        if inter_poly.geom_type == 'Polygon' and inter_poly.area > limiar:
                            delete_lines.append(z)
                            pols.at[z, 'union'] = i

            if pols.loc[i, 'union'] == 0:
                pols.at[i, 'union'] = i

        pols = pols.dissolve(by='union')
        pols.to_file(os.path.join(shape_path_aux,'test_union.geojson'), driver='GeoJSON')


    def join_vectors_2(self, shape_path_aux, limiar=0.1):

        result_paths = []
        for filename in os.listdir(shape_path_aux):
            if os.path.splitext(filename)[1].lower() == '.geojson' and \
                    os.path.splitext(filename)[0].lower() != 'final_result':
                result_paths.append(os.path.join(shape_path_aux, filename))

        pols = pd.concat([gpd.read_file(i) for i in result_paths], axis=0).reset_index(drop=True)
        pols['ID'] = pols.index
        pols.insert(1, column='union', value=0)

        delete_lines = []
        for i in pols.index:
            if pols.loc[i, 'union'] == 0:
                polygon = pols.loc[i, 'geometry']
                intersec = pols.overlaps(polygon)
                intersec = intersec[intersec]
                list_result = list(intersec.index)

                if len(list_result) != 0:
                    maior = pols.loc[list_result, 'geometry'].area.idxmax(axis=0)

                    for z in list_result:
                        inter_poly = pols.loc[z, 'geometry'].intersection(pols.loc[maior, 'geometry'])
                        if inter_poly.geom_type == 'Polygon' and inter_poly.area > limiar:
                            delete_lines.append(z)
                            pols.at[z, 'union'] = i

            if pols.loc[i, 'union'] == 0:
                pols.at[i, 'union'] = i

        pols = pols.dissolve(by='union')
        pols.to_file(os.path.join(shape_path_aux,'test_union.geojson'), driver='GeoJSON')

if __name__ == '__main__':
    print('teste')

# def get_iou(a, b, epsilon=1e-5):
#     """ Given two boxes aandb` defined as a list of four numbers:
#     [x1,y1,x2,y2]
#     where:
#     x1,y1 represent the upper left corner
#     x2,y2 represent the lower right corner
#     It returns the Intersect of Union score for these two boxes.
#
#     Args:
#         a:          (list of 4 numbers) [x1,y1,x2,y2]
#         b:          (list of 4 numbers) [x1,y1,x2,y2]
#         epsilon:    (float) Small value to prevent division by zero
#
#     Returns:
#         (float) The Intersect of Union score.
#     """
#     # COORDINATES OF THE INTERSECTION BOX
#     x1 = max(a[0], b[0])
#     y1 = max(a[1], b[1])
#     x2 = min(a[2], b[2])
#     y2 = min(a[3], b[3])
#
#     # AREA OF OVERLAP - Area where the boxes intersect
#     width = (x2 - x1)
#     height = (y2 - y1)
#     # handle case where there is NO overlap
#     if (width<0) or (height <0):
#         return 0.0
#     area_overlap = width * height
#
#     # COMBINED AREA
#     area_a = (a[2] - a[0]) * (a[3] - a[1])
#     area_b = (b[2] - b[0]) * (b[3] - b[1])
#     area_combined = area_a + area_b - area_overlap
#
#     # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#     iou = area_overlap / (area_combined+epsilon)
#     return iou
#
# def gt_pred_lists(gt_class_ids, gt_bboxes, pred_class_ids, pred_bboxes, iou_tresh=0.5):
#     """
#     Given a list of gt and predicted classes and their boxes,
#     this function associates the predicted classes to their gt classes using a given Iou (Iou>= 0.5 for example) and returns
#     two normalized lists of len = N containing the gt and predicted classes,
#     filling the non-predicted and miss-predicted classes by the background instance (index 0).
#
#     Args    :
#         gt_class_ids   :    list of gt classes of size N1
#         pred_class_ids :    list of predicted classes of size N2
#         gt_bboxes      :    list of gt boxes [N1, (x1, y1, x2, y2)]
#         pred_bboxes    :    list of pred boxes [N2, (x1, y1, x2, y2)]
#
#     Returns :
#         gt             :    list of size N
#         pred           :    list of size N
#
#     """
#
#     # dict containing the state of each gt and predicted class (0 : not associated to any other class, 1 : associated to a classe)
#     gt_class_ids_ = {'state': [0 * i for i in range(len(gt_class_ids))], "gt_class_ids": list(gt_class_ids)}
#     pred_class_ids_ = {'state': [0 * i for i in range(len(pred_class_ids))], "pred_class_ids": list(pred_class_ids)}
#
#     # the two lists to be returned
#     pred = []
#     gt = []
#
#     for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
#         for j, pred_class in enumerate(pred_class_ids_['pred_class_ids']):
#             # check if the gt object is overlapping with a predicted object
#             if get_iou(gt_bboxes[i], pred_bboxes[j]) >= iou_tresh:
#                 # change the state of the gt and predicted class when an overlapping is found
#                 gt_class_ids_['state'][i] = 1
#                 pred_class_ids_['state'][j] = 1
#                 # chack if the overlapping objects are from the same class
#                 if (gt_class == pred_class):
#                     gt.append(gt_class)
#                     pred.append(pred_class)
#                 # if the overlapping objects are not from the same class
#                 else:
#                     gt.append(gt_class)
#                     pred.append(pred_class)
#     # look for objects that are not predicted (gt objects that dont exists in pred objects)
#     for i, gt_class in enumerate(gt_class_ids_["gt_class_ids"]):
#         if gt_class_ids_['state'][i] == 0:
#             gt.append(gt_class)
#             pred.append(0)
#             # match_id += 1
#     # look for objects that are mispredicted (pred objects that dont exists in gt objects)
#     for j, pred_class in enumerate(pred_class_ids_["pred_class_ids"]):
#         if pred_class_ids_['state'][j] == 0:
#             gt.append(0)
#             pred.append(pred_class)
#     return gt, pred
