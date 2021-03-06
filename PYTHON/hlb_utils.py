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
import json

gdal.UseExceptions()


class PreProcessing:

    def __init__(self, rgb_dir, dem_dir, canopy_dir=None, proj_name=None):

        self.rgb_dir = rgb_dir
        self.dem_dir = dem_dir

        if proj_name is None:
            self.proj_name = os.path.splitext(os.path.basename(rgb_dir))[0].lower()
        else:
            self.proj_name = proj_name

        self.proj_dir = os.path.join('../datasets', self.proj_name)
        if not os.path.isdir(self.proj_dir):
            os.mkdir(self.proj_dir)
            os.mkdir(os.path.join(self.proj_dir, 'img_patches'))
            os.mkdir(os.path.join(self.proj_dir, 'annotations'))

        self.grid = None
        if canopy_dir is None:
            self.copas = None
        else:
            self.copas = gpd.read_file(canopy_dir).set_index('id').sort_index()

    def merge_rgb_dem(self):
        print('Reading source image information')
        with rasterio.open(self.rgb_dir) as src_rgb:
            meta_dst = src_rgb.meta
        meta_dst.update(count=4)

        folder_dir = os.path.dirname(self.rgb_dir)
        # # Read each layer and write it to stack
        print('Writing output file')
        with rasterio.open(os.path.join(folder_dir, 'stack_prev.tif'), 'w',
                           **meta_dst) as dst:
            for band in range(1, 4):
                print('Band {}'.format(band))
                with rasterio.open(self.rgb_dir) as src_rgb:
                    dst.write_band(band, src_rgb.read(band))

            with rasterio.open(self.dem_dir) as src_dem:
                print('Band 4')
                meta_dem = src_dem.meta

                band_dem = src_dem.read(1)
                band_dem[band_dem == meta_dem['nodata']] = np.nan
                min_value, max_value = np.nanmin(band_dem), np.nanmax(band_dem)

                band_dem_rescaled = np.interp(band_dem, (min_value, max_value), (0, 255))

                band_dem_rescaled[band_dem_rescaled == np.nan] = 0

                dst.write_band(4, band_dem_rescaled.astype('uint8'))

        print('Setting options to output file')
        ds = gdal.Open(os.path.join(folder_dir, 'stack_prev.tif'))
        gdal.Translate(os.path.join(folder_dir, 'RGB-D_0-255_stack.tif'), ds,
                       creationOptions=['ALPHA=NO'])
        ds = None
        os.remove(os.path.join(folder_dir, 'stack_prev.tif'))


        self.orto = gdal.Open(os.path.join(folder_dir, 'RGB-D_0-255_stack.tif'))  # Dataset da imagem
        self.pxsize = round(self.orto.GetGeoTransform()[1], 5)  # Tamanho do pixel (m)
        print('Done!')

    def split_img(self, patch_size=256, overlap=0, driver_grid='GeoJSON', save_img=True):
        self.patch_size = patch_size
        # Acessar EPSG do mosaico
        prj = self.orto.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        epsg = 'EPSG:' + srs.GetAuthorityCode('projcs')

        # Start do ID e do dicionario de poligonos formadores da grade
        i = 1
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

                    proj = osr.SpatialReference(wkt=ds.GetProjection())
                    src = 'EPSG:' + proj.GetAttrValue('AUTHORITY', 1)

                    # Caso seja declarado o arquivo de copas (para treinamento), cortar apenas os patches que cont??m
                    # algum vetor no seu interior (condi????o do if)
                    if self.copas is not None:
                        # Avaliar se o patch corresponde a uma regi??o da imagem com valores v??lidos (N??o zero)
                        # if len(ds.ReadAsArray().nonzero()[0]) != 0 and self.copas.intersects(geom).any():
                        overlayer = gpd.overlay(self.copas,
                                                gpd.GeoDataFrame({'geometry': gpd.GeoSeries(geom,
                                                                                            crs=src)}),
                                                how='intersection')

                        overlayer = overlayer.loc[overlayer['geometry'].geom_type == 'Polygon']

                        if len(ds.ReadAsArray().nonzero()[0]) != 0 and overlayer.any(axis=None):
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
                    # Caso n??o seja declarado o arquivo de copas (apenas para infer??ncia)
                    else:
                        # Avaliar se o patch corresponde a uma regi??o da imagem com valores v??lidos (N??o zero)
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

        # Gerar o GeoDataFrame com a grade e salv??-la
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
            # A fun????o .coords e .exterior n??o se aplicam a multipartes [FEITO]
            coord = list(corte.loc[i, 'geometry'].exterior.coords)

            segmentation = ''
            # Aten????o aqui!! A visualiza????o das annotations indicou que na lista de coors_img, o y precede o x
            for xy in coord:
                segmentation += str(round(img.index(xy[0], xy[1], float)[1], 2)) + ', ' + str(
                    round(img.index(xy[0], xy[1], float)[0], 2)) + ', '

            segmentation = segmentation[:-2]
            corte.at[i, 'geometry_image'] = segmentation

        return corte

    # def is_geom_valid(self, corte, geom='MultiPolygon'):
    #     '''
    #     Criar uma grade com patches selecionados
    #
    #     :param
    #     :return:
    #     '''
    #
    #     features = []
    #     for i in corte.index:
    #         if corte.loc[i, 'geometry'].geom_type == geom:
    #             features.append(i)
    #
    #     if len(features) != 0:
    #         return features
    #     else:
    #         return True

    # def check_geom_split(self):
    #     dict_invalid = {}
    #
    #     with tqdm(total=len(self.grid.index)) as pbar:
    #         pbar.set_description("Avaliando geometrias")
    #         sleep(0.1)
    #         for i in self.grid.index:
    #             # corte = gpd.clip(self.copas, self.grid.loc[[i]])
    #             corte = gpd.overlay(self.copas, self.grid.loc[[i]], how='intersection') # gpd.clip(self.copas, self.grid.loc[[i]]).explode().reset_index(drop=True, level=[1]).set_index('id')
    #             corte = corte.loc[corte['geometry'].geom_type == 'Polygon']
    #
    #             if not corte.empty and type(self.is_geom_valid(corte)) == list:
    #                     dict_invalid['{i:05}'.format(i=i)] = self.is_geom_valid(corte)
    #             pbar.update(1)
    #     return dict_invalid

    def split_vector(self):

        # dict_invalid = self.check_geom_split()

        # if len(dict_invalid) == 0:
        with tqdm(total=len(self.grid.index)) as pbar:
            pbar.set_description("Salvando arquivos")
            sleep(0.1)
            for i in self.grid.index:
                # Avaliar situa????o em que keep_geom (argumento de gpd.clip()) seja True
                corte = gpd.overlay(self.copas, self.grid.loc[[i]],
                                    how='intersection')  # gpd.clip(self.copas, self.grid.loc[[i]]).explode().reset_index(drop=True, level=[1]).set_index('id')
                corte = corte.loc[corte['geometry'].geom_type == 'Polygon']
                if not corte.empty:
                    corte = self.coords_to_xy(corte, self.grid.loc[i].name)
                    dir_out = os.path.join(self.proj_dir, 'vector_split.gpkg')
                    corte.to_file(dir_out, layer='{i:05}'.format(i=i), driver="GPKG")
                pbar.update(1)
            print('Conclu??do!')
        # else:
        #     print('Voc?? deve editar os vetores dos seguintes patches para que n??o sejam gerados poligonos multipartes', dict_invalid)
        #     return

    # def is_all_same_geom(self, geom='MultiPolygon'):
    #     '''
    #     Criar uma grade com patches selecionados
    #
    #     :param
    #     :return:
    #     '''
    #
    #     # grid_selected = gpd.read_file(grade_dir).set_index('id')
    #
    #     multi = {}
    #     for z in self.grid.index:
    #
    #         corte = gpd.read_file(os.path.join(self.proj_dir, 'vector_split.gpkg'), layer='{z:05}'.format(z=z))
    #         features = []
    #         for i in range(corte.shape[0]):
    #
    #             if corte.loc[i, 'geometry'].geom_type == geom:
    #                 features.append(i)
    #
    #         if len(features) != 0:
    #             multi['{z:05}'.format(z=z)] = features
    #
    #     if len(multi) == 0:
    #         return True
    #     else:
    #         return False, multi

    def get_bbox(self, geom):
        pairs = []

        for i in range(0, len(geom) - 1, 2):
            pairs.append((geom[i], geom[i + 1]))

        x = []
        y = []
        for i in pairs:
            x.append(i[0])
            y.append(i[1])

        bbox = [min(x), min(y), round(max(x) - min(x), 2), round(max(y) - min(y), 2)]

        return bbox

    def make_annotations(self, info=None, licenses=None, categories=None):

        if info is None:
            info = {"description": "Felipe Sa 2021 - HLB",
                    "url": "http://siteaqui.com",
                    "version": "1.0",
                    "year": 2021,
                    "contributor": "Felipe Sa",
                    "date_created": "2021/01/01"}
        if licenses is None:
            licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                         "id": 1,
                         "name": "Attribution-NonCommercial-ShareAlike License"},
                        {"url": "http://creativecommons.org/licenses/by-nc/2.0/",
                         "id": 2,
                         "name": "Attribution-NonCommercial License"}]
        if categories is None:
            categories = [{"supercategory": "canopy", "id": 1, "name": "hamlin"}]

        for split in ['train', 'val']:
            ids_img = list(self.grid.loc[self.grid['split_samples'] == split].index)
            images = []
            annotations = []
            for i in ids_img:
                # images
                images.append({"license": 1,
                               "file_name": "{:05}.tif".format(i),
                               "coco_url": "empty",
                               "height": self.patch_size,
                               "width": self.patch_size,
                               "date_captured": "2020-01-01 00:00:00",
                               "flickr_url": "empty",
                               "id": i
                               })

                # Annotations
                copas = gpd.read_file(os.path.join(self.proj_dir, 'vector_split.gpkg'), layer='{i:05}'.format(i=i))

                for row in copas.index:
                    geom_px = [float(x) for x in copas.loc[row, 'geometry_image'].split(', ')]

                    annotations.append({"segmentation": [geom_px],
                                        # [REFATORADO] Aten????o aqui!! 0.0025 ?? para o tamanho do pixel de 5cm (0,05 x 0,05 = 0,0025)
                                        "area": copas.loc[row, 'geometry'].area / (self.pxsize * self.pxsize),
                                        "iscrowd": 0,
                                        "image_id": i,
                                        "bbox": self.get_bbox(geom_px),
                                        "category_id": 1,
                                        "id": int(str(i) + '0' + str(row + 1))  # N??o definido ainda
                                        })

            annotation = {
                "info": info,
                "licenses": licenses,
                "images": images,
                "categories": categories,
                "annotations": annotations,  # <-- Not in Captions annotations
                # "segment_info": []  # <-- Only in Panoptic annotations
            }

            with open(os.path.join(self.proj_dir, 'annotations', 'coco_annotations_{}.json'.format(split)),
                      'w') as outfile:
                json.dump(annotation, outfile)

    def make_annotations_joined(self, dict,  info=None, licenses=None, categories=None):

        if info is None:
            info = {"description": "Felipe Sa 2021 - HLB",
                    "url": "http://siteaqui.com",
                    "version": "1.0",
                    "year": 2021,
                    "contributor": "Felipe Sa",
                    "date_created": "2021/01/01"}
        if licenses is None:
            licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                         "id": 1,
                         "name": "Attribution-NonCommercial-ShareAlike License"},
                        {"url": "http://creativecommons.org/licenses/by-nc/2.0/",
                         "id": 2,
                         "name": "Attribution-NonCommercial License"}]
        if categories is None:
            categories = [{"supercategory": "canopy", "id": 1, "name": "hamlin"}]

        for split in ['train', 'val']:
            ids_img = list(self.grid.loc[self.grid['split_samples'] == split].index)
            images = []
            annotations = []
            for i in ids_img:

                for k, v in dict.items():
                    if i in k:
                        self.patch_size = v[0]
                        self.pxsize = v[1]

                # images
                images.append({"license": 1,
                               "file_name": "{:05}.tif".format(i),
                               "coco_url": "empty",
                               "height": self.patch_size,
                               "width": self.patch_size,
                               "date_captured": "2020-01-01 00:00:00",
                               "flickr_url": "empty",
                               "id": i
                               })

                # Annotations
                copas = gpd.read_file(os.path.join(self.proj_dir, 'vector_split.gpkg'), layer='{i:05}'.format(i=i))

                for row in copas.index:
                    geom_px = [float(x) for x in copas.loc[row, 'geometry_image'].split(', ')]

                    annotations.append({"segmentation": [geom_px],
                                        # [REFATORADO] Aten????o aqui!! 0.0025 ?? para o tamanho do pixel de 5cm (0,05 x 0,05 = 0,0025)
                                        "area": copas.loc[row, 'geometry'].area / (self.pxsize * self.pxsize),
                                        "iscrowd": 0,
                                        "image_id": i,
                                        "bbox": self.get_bbox(geom_px),
                                        "category_id": 1,
                                        "id": int(str(i) + '0' + str(row + 1))  # N??o definido ainda
                                        })

            annotation = {
                "info": info,
                "licenses": licenses,
                "images": images,
                "categories": categories,
                "annotations": annotations,  # <-- Not in Captions annotations
                # "segment_info": []  # <-- Only in Panoptic annotations
            }

            with open(os.path.join(self.proj_dir, 'annotations', 'coco_annotations_{}.json'.format(split)),
                      'w') as outfile:
                json.dump(annotation, outfile)


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

        return 'Vector created!!!!'

    def result_to_vector(self, img_tif, masks, scores):

        gt = img_tif.GetGeoTransform()
        proj = osr.SpatialReference(wkt=img_tif.GetProjection())
        src = 'EPSG:' + proj.GetAttrValue('AUTHORITY', 1)
        polygons = []

        serie = gpd.GeoSeries(crs=src)
        gdf_final = gpd.GeoDataFrame({'score': 0.00, 'geometry': serie})

        for i_mask, score in zip(range(masks.shape[2]), scores):
            x, y = np.where(masks[:, :, i_mask] == True)

            coord_x = (y * gt[1]) + gt[0]
            coord_y = (x * gt[5]) + gt[3]

            poly_coord = [[x, y] for x, y in zip(coord_x, coord_y)]
            # poly_coord = []
            # for x, y in zip(coord_x, coord_y):
            #     poly_coord.append((x, y))

            # Por causa do Bug em 15/10
            if len(poly_coord) < 0:
                poly_coord.append(poly_coord[0])

            # Por causa do bug em 08/09
            if len(poly_coord) >= 3:
                #poly = geometry.Polygon(poly_coord).buffer(0.15, join_style=3).buffer(-0.15, join_style=3)

                poly = geometry.MultiPoint(poly_coord).convex_hull
                gdf_final = gdf_final.append({'geometry': poly, 'detection_score': score}, ignore_index=True)


        gdf_final = gdf_final.explode().reset_index(drop=True)
        return gdf_final

    def join_vectors(self, shape_path_aux, centroid_dist=1, limiar_ovelap=0.6):

        result_paths = []
        for filename in os.listdir(shape_path_aux):
            if os.path.splitext(filename)[1].lower() == '.geojson':
                result_paths.append(os.path.join(shape_path_aux, filename))

        pols = pd.concat([gpd.read_file(i) for i in result_paths], axis=0).reset_index(drop=True)
        pols = pols[pols.geometry.is_valid]
        pols['geometry'] = pols.buffer(0.1).buffer(-0.1).simplify(0.03)
        pols = pols.explode('geometry').reset_index(drop=True)

        '''
        pols['centroid'] = pols['geometry'].centroid
        pols.insert(1, column='union', value=0)
        pols.set_index(keys=pd.Index(range(1, pols.shape[0] + 1)), inplace=True)

        with tqdm(total=len(pols.index)) as pbar:
            pbar.set_description("Simplificando - Macro")
            sleep(0.1)
            for i in pols.index:
                if pols.loc[i, 'union'] == 0:
                    point = pols.loc[i, 'centroid']
                    distance = pols['centroid'].distance(point)

                    distance = distance[distance <= centroid_dist]

                    if len(distance) > 1:
                        pols.at[distance.index, 'union'] = i
                    else:
                        pols.at[i, 'union'] = i
                pbar.update(1)

        # pols.drop('centroid', axis=1, inplace=True)
        pols = pols.dissolve(by='union', as_index=False)
        pols.drop(['union', 'centroid'], axis=1, inplace=True)
        pols.drop(pols.loc[pols['geometry'].is_empty].index, axis=0, inplace=True)
        '''
        pols.set_index(keys=pd.Index(range(1, pols.shape[0] + 1)), inplace=True)
        pols['id'] = pols.index
        # pols['centroid'] = pols['geometry'].centroid

        for _ in range(2):
            # Gerar intersecoes entre os poligonos encontrados
            # Uso de busca indexada (R-tree) com a fun????o gpd.overlay()
            over = gpd.overlay(pols, pols, how='intersection')
            # Remover poligonos vazios (talvez possa ser dispensado a partir de agora)
            over = over.loc[-over['geometry'].is_empty]
            # Remover intersecoes originadas pela sobreposi????o da fei????o com ela mesma
            over = over.loc[over['id_1'] != over['id_2']]
            # .explode().reset_index(drop=True)

            delete_lines = []
            # Barra de progresso #
            with tqdm(total=len(over['id_1'])) as pbar:
                pbar.set_description("Simplificando - Micro")
                sleep(0.1)

                over['area'] = over['geometry'].area
                over = over.sort_values('area')

                # Para cada poligono..
                for i in over['id_1']:
                    # intersec????es entre o poligono avaliado e todas as outras fei????es
                    intersec = over[over['id_1'] == i]

                    # Caso alguma ??rea de intersecao avaliada seja maior que 60% (limiar_overlap) do proprio poligono avaliado
                    # ou se poligono avaliado ?? 3x menor que aquele com o qual existe intersecao, o poligono ser?? excluido]
                    if intersec.loc[intersec['geometry'].area >= limiar_ovelap * pols.loc[i, 'geometry'].area].any(
                            axis=None) | \
                            (intersec['geometry'].area.sum() >= limiar_ovelap * pols.loc[i, 'geometry'].area) | \
                            (3 * pols.loc[i, 'geometry'].area < pols.loc[intersec['id_2'], 'geometry'].area).any():

                        # Inserido em 18/12 para evitar falha de detec????o
                        # N??o deletar o maior poligono formador da maior intersecao
                        maior = intersec.loc[
                            intersec['geometry'].area == max(intersec['geometry'].area), ['id_1', 'id_2']]
                        if pols.loc[maior['id_1'], 'geometry'].area.values[0] < \
                                pols.loc[maior['id_2'], 'geometry'].area.values[0]:
                            delete_lines.append(i)

                    pbar.update(1)

                # Identificar poligonos a serem apagados
                delete_lines = (list(set(delete_lines)))
            # Fim da barra de progresso #

            pols.drop(labels=delete_lines, axis=0, inplace=True)

        pols = pols.explode().reset_index(drop=True)

        pols.set_index(keys=pd.Index(range(1, pols.shape[0] + 1)), inplace=True)
        pols['id'] = pols.index

        pols_filtred = pols.loc[pols['detection_score'] >= 0.99]

        return pols, pols_filtred


if __name__ == '__main__':
    print('teste')

# def get_iou(a, b, epsilon=1e-5):
#     """ Given two boxes a and b` defined as a list of four numbers:
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
