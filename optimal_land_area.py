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
import hashlib
import logging
import os
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

gdal.SetCacheMax(2**27)

BASE_DATA_DIR = 'data'
WORKSPACE_DIR = 'workspace_dir'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
CLIPPED_DIR = os.path.join(CHURN_DIR, 'clipped')
DOWNLOAD_DIR = os.path.join(CHURN_DIR, 'downloads')
COUNTRY_WORKSPACES = os.path.join(CHURN_DIR, 'country_workspaces')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'output')
RASTER_SUBSET_LIST = [
    'realized_coastalprotection',
    'realized_natureaccess10_nathab',
    'realized_nitrogenretention_nathab_clamped',
    'realized_pollination_nathab_clamped',
    'realized_sedimentdeposition_nathab_clamped',
]

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
    try:
        os.makedirs(target_dir)
    except OSError:
        pass
    subprocess.run(
        f'gsutil cp -r "{gs_uri}/*" "{target_dir}"',
        shell=True, check=True)
    with open(token_file_path, 'w') as token_file:
        token_file.write("done")


def extract_feature(
        base_vector_path, base_fieldname, base_fieldname_value,
        target_vector_path):
    """Extract a feature from base into a new vector.

    Args:
        base_vector_path (str): path to a multipolygon vector
        base_fieldname (str): name of fieldname to filter on
        base_fieldname_value (str): value of fieldname to filter on
        target_vector_path (str): path to target vector.

    Returns:
        None.

    """
    try:
        os.makedirs(os.path.dirname(target_vector_path))
    except OSError:
        pass
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    layer.SetAttributeFilter(f"{base_fieldname}='{base_fieldname_value}'")
    country_feature = next(iter(layer))
    country_geometry = country_feature.GetGeometryRef()
    gpkg_driver = ogr.GetDriverByName('GPKG')

    local_country_vector = gpkg_driver.CreateDataSource(target_vector_path)
    # create the layer
    local_layer = local_country_vector.CreateLayer(
        os.path.basename(os.path.splitext(base_fieldname_value)[0]),
        layer.GetSpatialRef(), ogr.wkbMultiPolygon)
    layer_defn = local_layer.GetLayerDefn()
    new_feature = ogr.Feature(layer_defn)
    new_feature.SetGeometry(country_geometry.Clone())
    country_geometry = None
    local_layer.CreateFeature(new_feature)
    new_feature = None
    local_layer = None
    local_country_vector = None


def main():
    """Entry point."""
    # convert raster list to just 1-10 integer
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, CLIPPED_DIR, OUTPUT_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    bucket_uri = (
        'gs://critical-natural-capital-ecoshards/realized_service_ecoshards/'
        'by_country')
    m = hashlib.md5()
    m.update(bucket_uri)
    local_churn_dir = os.path.join(CHURN_DIR, m.hexdigest())
    local_download_dir = os.path.join(local_churn_dir, 'downloads')
    token_file = os.path.join(
        local_download_dir, f'{os.path.basename(bucket_uri)}.token')
    task_graph = taskgraph.TaskGraph(local_churn_dir, -1, 5.0)
    copy_gs_task = task_graph.add_task(
        func=copy_gs,
        args=(bucket_uri, local_download_dir, token_file),
        target_path_list=[token_file])
    copy_gs_task.join()

    # we know there's a "countries" .gpkg in there
    global_vector_path = glob.glob(
        os.path.join(local_download_dir, 'countries*.gpkg'))[0]

    base_raster_path_list = [
        path for path in glob.glob(os.path.join(local_download_dir, '*.tif'))
        if any(x in path for x in RASTER_SUBSET_LIST)]

    clipped_pixel_length = min([
        pygeoprocessing.get_raster_info(path)['pixel_size'][0]
        for path in base_raster_path_list])

    for country_iso in ['IND']:
        country_working_dir = os.path.join(
            local_churn_dir, 'country_workspaces', country_iso)
        try:
            os.makedirs(country_working_dir)
        except OSError:
            pass

        local_country_vector_path = os.path.join(
            country_working_dir, f'{country_iso}.gpkg')
        extract_task = task_graph.add_task(
            func=extract_feature,
            args=(global_vector_path, 'iso3', country_iso,
                  local_country_vector_path),
            hash_target_files=False,
            target_path_list=[local_country_vector_path],
            task_name=f'extract {country_iso}')

        clipped_raster_path_list = [
            os.path.join(
                country_working_dir, 'clipped', os.path.basename(raster_path))
            for raster_path in base_raster_path_list]

        align_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(
                base_raster_path_list, clipped_raster_path_list,
                ['near'] * len(clipped_raster_path_list),
                [clipped_pixel_length, -clipped_pixel_length],
                # 'intersection',
                [80, 20, 81, 21], # area in mid india
                ),
            kwargs={
                'base_vector_path_list': [local_country_vector_path],
                'vector_mask_options': {
                    'mask_vector_path': local_country_vector_path,
                }
            },
            dependent_task_list=[extract_task],
            ignore_path_list=[local_country_vector_path],
            target_path_list=clipped_raster_path_list,
            task_name=f'clip align task for {country_iso}')

        target_suffix = country_iso
        logging.getLogger('pygeoprocessing').setLevel(logging.DEBUG)
        local_output_dir = os.path.join(local_churn_dir, 'output')
        task_graph.add_task(
            func=pygeoprocessing.raster_optimization,
            args=(
                [(x, 1) for x in clipped_raster_path_list],
                local_churn_dir, local_output_dir),
            kwargs={'target_suffix': target_suffix},
            dependent_task_list=[align_task],
            task_name=f'optimize {country_iso}')

    task_graph.join()
    task_graph.close()
    return

    # TODO: these show how to smooth the final raster val if desired
    # optimal_raw_mask_path = os.path.join(
    #     OUTPUT_DIR, f'optimal_mask_{target_suffix}.tif')

    # smoothed_mask_path = os.path.join(
    #     OUTPUT_DIR, f'smoothed_mask_{target_suffix}.tif')

    # smooth_radius = 3
    # smooth_mask(optimal_raw_mask_path, smooth_radius, smoothed_mask_path)


if __name__ == '__main__':
    main()
