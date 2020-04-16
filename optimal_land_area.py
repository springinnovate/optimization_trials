"""Try to get minimum land with no less than 90% of service.

Here's the plan:
    * Smooth the rasters w/ a gaussian blur first to flatten out any sharp
      pixels.
    * For each variable raster, make a connected components raster (polygon?)
    * Then make a greatest combined marginal value polygon
        * Everywhere connected components overlap, make that another polygon
        * i.e. this polygon is where all the values are the same.
    * For connected components:
        * Must be in Cython...
        * Go by iterblocks
            * Look for pixel that is not connected and start search
            * Grow out connected component until no other touching pixels.
        * Polygonalize by making a point in every pixel center.

"""
import glob
import logging
import os
import math
import subprocess
import sys
import tempfile

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

BASE_DATA_DIR = 'data'
WORKSPACE_DIR = 'workspace_dir'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
CLIPPED_DIR = os.path.join(CHURN_DIR, 'clipped')
COUNTRY_WORKSPACES = os.path.join(CHURN_DIR, 'country_workspaces')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'output')
TARGET_NODATA = -1
PROP_NODATA = -1
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


def sum_raster(raster_path_band):
    """Sum the raster and return the result."""
    nodata = pygeoprocessing.get_raster_info(
        raster_path_band[0])['nodata'][raster_path_band[1]-1]

    raster_sum = 0.0
    for _, array in pygeoprocessing.iterblocks(raster_path_band):
        valid_mask = ~numpy.isclose(array, nodata)
        raster_sum += numpy.sum(array[valid_mask])

    return raster_sum


def make_neighborhood_hat_kernel(kernel_size, kernel_filepath):
    """Make a hat kernel that's the sum in the center and 1 <= kernel_size.

    Args:
        kernel_size (int): kernel should be kernel_size X kernel_size
        kernel_filepath (str): path to target kernel.

    Returns:
        None

    """
    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath, kernel_size, kernel_size, 1, gdal.GDT_Float32)
    kernel_raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_raster.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    kernel_array = (numpy.sqrt(numpy.sum(
        [(index - kernel_size//2)**2
         for index in numpy.meshgrid(
            range(kernel_size), range(kernel_size))], axis=0)) <=
        kernel_size//2).astype(numpy.uint8)

    # make the center the sum of the area of the circle so it's always on
    kernel_array[kernel_array//2, kernel_array//2] = (
        3.14159 * (kernel_size//2+1)**2)
    kernel_band.WriteArray(kernel_array)
    kernel_band = None
    kernel_raster = None


def smooth_mask(base_mask_path, smooth_radius, target_smooth_mask_path):
    """Fill in gaps in base mask if there are neighbors.

    Args:
        base_mask_path (str): path to base raster, should be 0, 1 and nodata.
        smooth_radius (int): how far to smooth out at a max radius?
        target_smooth_mask_path (str): target smoothed file.

    Returns:
        None.

    """
    kernel_size = smooth_radius*2+1
    working_dir = tempfile.mkdtemp(
        dir=os.path.dirname(target_smooth_mask_path))
    kernel_path = os.path.join(working_dir, f'kernel_{kernel_size}.tif')
    make_neighborhood_hat_kernel(kernel_size, kernel_path)

    convolved_raster_path = os.path.join(working_dir, 'convolved_mask.tif')
    byte_nodata = 255
    pygeoprocessing.convolve_2d(
        (base_mask_path, 1), (kernel_path, 1), convolved_raster_path,
        ignore_nodata=False, working_dir=working_dir, mask_nodata=False,
        target_nodata=TARGET_NODATA)

    # set required proportion of coverage to turn on a pixel, lets make it a
    # quarter wedge.
    proportion_covered = 0.01
    threshold_val = proportion_covered * 3.14159 * (smooth_radius+1)**2
    pygeoprocessing.raster_calculator(
        [(convolved_raster_path, 1), (threshold_val, 'raw'),
         (TARGET_NODATA, 'raw'), (byte_nodata, 'raw'), ], threshold_op,
        target_smooth_mask_path, gdal.GDT_Byte, byte_nodata)

    # try:
    #     shutil.rmtree(working_dir)
    # except OSError:
    #     LOGGER.warn("couldn't delete %s", working_dir)
    #     pass


def threshold_op(base_array, threshold_val, base_nodata, target_nodata):
    """Threshold base to 1 where val >= threshold_val."""
    result = numpy.empty(base_array.shape, dtype=numpy.uint8)
    result[:] = target_nodata
    valid_mask = ~numpy.isclose(base_array, base_nodata) & (
        ~numpy.isclose(base_array, 0))
    result[valid_mask] = base_array[valid_mask] >= threshold_val
    return result


def proportion_op(base_array, total_sum, base_nodata, target_nodata):
    """Divide base by total and guard against nodata."""
    result = numpy.empty(base_array.shape, dtype=numpy.float64)
    result[:] = target_nodata
    valid_mask = ~numpy.isclose(base_array, base_nodata)
    result[valid_mask] = (
        base_array[valid_mask].astype(numpy.float64) / total_sum)
    return result


def sum_rasters_op(nodata, *array_list):
    """Sum all non-nodata pixels in array_list."""
    result = numpy.zeros(array_list[0].shape)
    total_valid_mask = numpy.zeros(array_list[0].shape, dtype=numpy.bool)
    for array in array_list:
        valid_mask = ~numpy.isclose(array, nodata)
        total_valid_mask |= valid_mask
        result[valid_mask] += array[valid_mask]
    result[~total_valid_mask] = nodata
    return result


def copy_gs(gs_uri, target_dir, token_file_path):
    """Copy uri dir to target and touch a token_file."""
    LOGGER.debug(' to copy %s to %s', gs_uri, target_dir)
    subprocess.run(
        f'gsutil cp -r "{gs_uri}/*" "{target_dir}"',
        shell=True, check=True)
    with open(token_file_path, 'w') as token_file:
        token_file.write("done")


def main():
    """Entry point."""
    # convert raster list to just 1-10 integer
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, CLIPPED_DIR, OUTPUT_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)

    bucket_uri = 'gs://critical-natural-capital-ecoshards/realized_service_ecoshards/by_country'
    token_file = os.path.join(
        CHURN_DIR, f'{os.path.basename(bucket_uri)}.token')
    copy_gs_task = task_graph.add_task(
        func=copy_gs,
        args=(bucket_uri, CHURN_DIR, token_file),
        target_path_list=[token_file])
    copy_gs_task.join()

    country_vector = gdal.OpenEx(
        glob.glob(os.path.join(CHURN_DIR, 'countries*.gpkg'))[0],
        gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()

    for country_iso in ['IND']:
        country_working_dir = os.path.join(COUNTRY_WORKSPACES, country_iso)
        try:
            os.makedirs(country_working_dir)
        except OSError:
            pass
        country_layer.ResetReading()
        country_layer.SetAttributeFilter(f"iso3='{country_iso}'")
        country_feature = next(iter(country_layer))
        LOGGER.debug(country_feature.GetField('iso3'))
        country_geometry = country_feature.GetGeometryRef()

        local_country_vector_path = os.path.join(
            country_working_dir, f'{country_iso}.gpkg')
        gpkg_driver = ogr.GetDriverByName('GPKG')

        LOGGER.debug('create vector')
        local_country_vector = gpkg_driver.CreateDataSource(
            local_country_vector_path)
        # create the layer
        LOGGER.debug('create layer')
        local_layer = local_country_vector.CreateLayer(
            country_iso, country_layer.GetSpatialRef(),
            ogr.wkbMultiPolygon)
        LOGGER.debug('get layer defn')
        layer_defn = local_layer.GetLayerDefn()
        LOGGER.debug('build feature')
        new_feature = ogr.Feature(layer_defn)
        LOGGER.debug('set geometry')
        new_feature.SetGeometry(country_geometry.Clone())
        country_geometry = None
        LOGGER.debug('create feature')
        local_layer.CreateFeature(new_feature)
        new_feature = None
        local_layer = None
        local_country_vector = None

        raster_path_list = [
            path for path in glob.glob(os.path.join(CHURN_DIR, '*.tif'))]
        clipped_raster_path_list = [
            os.path.join(
                country_working_dir, os.path.basename(raster_path))
            for raster_path in raster_path_list]

        target_pixel_length = min([
            pygeoprocessing.get_raster_info(path)['pixel_size'][0]
            for path in raster_path_list])

        LOGGER.debug(f'aligning rasters for {country_iso}')
        align_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(
                raster_path_list, clipped_raster_path_list,
                ['near'] * len(clipped_raster_path_list),
                [target_pixel_length, -target_pixel_length],
                'intersection',
                ),
            kwargs={
                'base_vector_path_list': [local_country_vector_path],
                'vector_mask_options': {
                    'mask_vector_path': local_country_vector_path,
                }
            },
            target_path_list=clipped_raster_path_list,
            task_name=f'clip align task for {country_iso}')

        target_suffix = country_iso
        logging.getLogger('pygeoprocessing').setLevel(logging.WARNING)
        pygeoprocessing.raster_optimization(
            [(x, 1) for x in clipped_raster_path_list], OUTPUT_DIR,
            target_suffix=target_suffix)
    task_graph.join()
    task_graph.close()
    del task_graph
    return

    # target
    optimal_raw_mask_path = os.path.join(
        OUTPUT_DIR, f'optimal_mask_{target_suffix}.tif')

    smoothed_mask_path = os.path.join(
        OUTPUT_DIR, f'smoothed_mask_{target_suffix}.tif')

    smooth_radius = 3
    smooth_mask(optimal_raw_mask_path, smooth_radius, smoothed_mask_path)


if __name__ == '__main__':
    main()
