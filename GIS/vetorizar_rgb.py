from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterRasterLayer
from qgis.core import QgsProcessingParameterVectorDestination
import processing


class Vetorizar_rgb(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer('RGB', 'RGB', defaultValue=None))
        self.addParameter(QgsProcessingParameterVectorDestination('Saida', 'saida', type=QgsProcessing.TypeVectorPolygon, createByDefault=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(7, model_feedback)
        results = {}
        outputs = {}

        # Raster calculator
        alg_params = {
            'CELLSIZE': 0,
            'CRS': None,
            'EXPRESSION': '(\"RGB@2\" - \"RGB@1\")/(\"RGB@2\" + \"RGB@1\" - \"RGB@3\")*10',
            'EXTENT': None,
            'LAYERS': parameters['RGB'],
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['RasterCalculator'] = processing.run('qgis:rastercalculator', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Smoothing
        alg_params = {
            'in': outputs['RasterCalculator']['OUTPUT'],
            'outputpixeltype': 5,
            'type': 'gaussian',
            'type.anidif.conductance': 1,
            'type.anidif.nbiter': 10,
            'type.anidif.timestep': 0.125,
            'type.gaussian.maxerror': 0.01,
            'type.gaussian.maxwidth': 32,
            'type.gaussian.stdev': 2,
            'type.mean.radius': 2,
            'out': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Smoothing'] = processing.run('otb:Smoothing', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # GrayScaleMorphologicalOperation
        alg_params = {
            'channel': 1,
            'filter': 'opening',
            'in': outputs['Smoothing']['out'],
            'outputpixeltype': 5,
            'structype': 'ball',
            'xradius': 3,
            'yradius': 3,
            'out': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Grayscalemorphologicaloperation'] = processing.run('otb:GrayScaleMorphologicalOperation', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        # r.reclass
        alg_params = {
            'GRASS_RASTER_FORMAT_META': '',
            'GRASS_RASTER_FORMAT_OPT': '',
            'GRASS_REGION_CELLSIZE_PARAMETER': 0,
            'GRASS_REGION_PARAMETER': None,
            'input': outputs['Grayscalemorphologicaloperation']['out'],
            'rules': '',
            'txtrules': '-16 thru 1.1 = 0\n* = *',
            'output': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Rreclass'] = processing.run('grass7:r.reclass', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}

        # GrayScaleMorphologicalOperation
        alg_params = {
            'channel': 1,
            'filter': 'opening',
            'in': outputs['Rreclass']['output'],
            'outputpixeltype': 5,
            'structype': 'ball',
            'xradius': 5,
            'yradius': 5,
            'out': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Grayscalemorphologicaloperation'] = processing.run('otb:GrayScaleMorphologicalOperation', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}

        # r.reclass
        alg_params = {
            'GRASS_RASTER_FORMAT_META': '',
            'GRASS_RASTER_FORMAT_OPT': '',
            'GRASS_REGION_CELLSIZE_PARAMETER': 0,
            'GRASS_REGION_PARAMETER': None,
            'input': outputs['Grayscalemorphologicaloperation']['out'],
            'rules': '',
            'txtrules': '1 thru 6 = 1\n0 = 0',
            'output': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['Rreclass'] = processing.run('grass7:r.reclass', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(6)
        if feedback.isCanceled():
            return {}

        # Polygonize (raster to vector)
        alg_params = {
            'BAND': 1,
            'EIGHT_CONNECTEDNESS': False,
            'EXTRA': '',
            'FIELD': 'DN',
            'INPUT': outputs['Rreclass']['output'],
            'OUTPUT': parameters['Saida']
        }
        outputs['PolygonizeRasterToVector'] = processing.run('gdal:polygonize', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Saida'] = outputs['PolygonizeRasterToVector']['OUTPUT']
        return results

    def name(self):
        return 'Vetorizar_RGB'

    def displayName(self):
        return 'Vetorizar_RGB'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Vetorizar_rgb()
