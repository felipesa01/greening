import os

import rasterio
import rasterio.plot
import rasterio.plot
from osgeo import gdal, osr
import shapely
from shapely import geometry
import geopandas as gpd

gdal.UseExceptions()


class GetPatches:

    def __init__(self, ortomosaico_dir, copas_dir, img_dir='../IMGS/CORTES', vector_dir='../VECTOR'):

        self.orto = gdal.Open(ortomosaico_dir)  # Dataset da imagem
        self.copas = gpd.read_file(copas_dir).set_index('id').sort_index()  # GeoDataFrame do arquivo vetorial
        self.img_dir = img_dir  # Diretório da pasta onde serão armazenados os patches
        self.vector_dir = vector_dir  # Diretório da pasta onde serão armazenados os vetores
        self.vector_ed_dir = os.path.join(vector_dir, 'EDITADOS')
        self.grid = None
        self.prefixo = ''

    def split_img(self, size=512, driver_grid='GeoJSON'):

        # Acessar EPSG do mosaico
        prj = self.orto.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        epsg = 'EPSG:' + srs.GetAuthorityCode('projcs')

        # Tamanho das imagens (px)
        patch_size = size

        # Start do ID e do dicionario de poligonos formadores da grade
        i = 1
        grid_pol = {'id': [], 'geometry': []}

        # Iterar em cada linha da imagem com passos de mesmo tamanho dos patches
        for px_x in range(0, self.orto.RasterXSize, patch_size):

            # Iterar em cada coluna da imagem com passos de mesmo tamanho dos patches
            for px_y in range(0, self.orto.RasterYSize, patch_size):

                # Gerar o dataset do patch na memoria
                ds = gdal.Translate('/vsimem/corte_{i}.tif'.format(i=i),
                                    self.orto,
                                    srcWin=[px_x, px_y, patch_size, patch_size])

                # Gerar poligono envolvente da imagem
                xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
                width, height = ds.RasterXSize, ds.RasterYSize
                xmax = xmin + width * xpixel
                ymin = ymax + height * ypixel
                geom = shapely.geometry.box(xmin, ymin, xmax, ymax)

                # Avaliar se o patch corresponde a uma região da imagem com valores válidos (Não zero) e
                # se o patch apresenta alguma copa no seu interior
                if len(ds.ReadAsArray().nonzero()[0]) != 0 and self.copas.intersects(geom).any():
                    # Salvar o pacth em disco
                    gdal.Translate(os.path.join(self.img_dir, 'corte_{i}.tif'.format(i=i)),
                                   self.orto,
                                   srcWin=[px_x, px_y, patch_size, patch_size])

                    # Inserir poligono na lista
                    grid_pol['id'].append(i)
                    grid_pol['geometry'].append(geom)

                    # Atualizar identificador
                    i += 1

        # Gerar o GeoDataFrame com a grade e salvá-la
        grid = gpd.GeoDataFrame(grid_pol, crs=epsg).set_index('id')
        self.grid = grid
        grid.to_file(os.path.join(self.vector_dir, 'grid_from_img.geojson'), driver=driver_grid)
        pass

    def is_same_size(self):
        pass

    def split_vector(self, prefixo=''):

        list_empty = []
        self.prefixo = prefixo

        for i in self.grid.index:
            corte = gpd.clip(self.copas, self.grid.loc[[i]])

            if corte.empty:
                list_empty.append(i)
            else:
                dir_out = os.path.join(self.vector_dir, 'CORTES', prefixo + '{}.geojson'.format(i))
                corte.to_file(dir_out, driver="GeoJSON")

        return list_empty

    def is_all_same_geom(self, geom='MultiPolygon'):
        '''
        Criar uma grade com patches selecionados

        :param grade_dir:
        :return:
        '''

        # grid_selected = gpd.read_file(grade_dir).set_index('id')

        multi = {}
        for z in self.grid.index:

            corte = gpd.read_file(os.path.join(self.vector_dir, 'CORTES', self.prefixo + '{}.geojson'.format(z)))

            corte.insert(len(corte.columns), 'geometry_image', 0)
            corte['geometry_image'] = corte['geometry_image'].astype('object')

            features = []
            for i in range(corte.shape[0]):

                if corte.loc[i, 'geometry'].geom_type == geom:
                    features.append(i)

            if len(features) != 0:
                multi[z] = features

        return multi

    def coords_to_xy(self, grid_selected):

        for z in grid_selected.index:

            img = rasterio.open('../IMGS/CORTES/corte_{}.tif'.format(z))
            corte = gpd.read_file('../VECTOR/CORTES/corte_{}.geojson'.format(z))

            corte.insert(len(corte.columns) - 1, 'geometry_image', 0)
            corte['geometry_image'] = corte['geometry_image'].astype('string')

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

            corte.to_file('../VECTOR/CORTES/EDITADOS/corte_{}.geojson'.format(z), driver="GeoJSON")


if __name__ == '__main__':
    print('teste')
