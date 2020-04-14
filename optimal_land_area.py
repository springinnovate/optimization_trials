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
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'output')
PROPORTIONAL_DIR = os.path.join(WORKSPACE_DIR, 'proportional_rasters')
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


def make_exponential_decay_kernel_raster(expected_distance, kernel_filepath):
    """Create a raster-based exponential decay kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Args:
        expected_distance (int or float): The distance (in pixels) of the
            kernel's radius, the distance at which the value of the decay
            function is equal to `1/e`.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None
    """
    max_distance = expected_distance * 5
    kernel_size = int(numpy.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName('GTiff')
    kernel_dataset = driver.Create(
        kernel_filepath.encode('utf-8'), kernel_size, kernel_size, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = numpy.indices((row_block_width,
                                                      col_block_width),
                                                     dtype=numpy.float)

            row_indices += numpy.float(row_offset - max_distance)
            col_indices += numpy.float(col_offset - max_distance)

            kernel_index_distances = numpy.hypot(
                row_indices, col_indices)
            kernel = numpy.where(
                kernel_index_distances > max_distance, 0.0,
                numpy.exp(-kernel_index_distances / expected_distance))
            integration += numpy.sum(kernel)

            kernel_band.WriteArray(kernel, xoff=col_offset,
                                   yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_band.FlushCache()
    kernel_dataset.FlushCache()

    for block_data in pygeoprocessing.iterblocks(
            (kernel_filepath, 1), offset_only=True):
        kernel_block = kernel_band.ReadAsArray(**block_data)
        kernel_block /= integration
        kernel_band.WriteArray(kernel_block, xoff=block_data['xoff'],
                               yoff=block_data['yoff'])

    kernel_band.FlushCache()
    kernel_dataset.FlushCache()
    kernel_band = None
    kernel_dataset = None


def clamp_to_integer(array, base_nodata, target_nodata):
    """Round the values in array to nearest integer."""
    result = numpy.empty(array.shape, dtype=numpy.uint8)
    result[:] = target_nodata
    valid_mask = ~numpy.isclose(array, base_nodata)
    result[valid_mask] = numpy.round(array[valid_mask]).astype(numpy.uint8)
    return result


def overlap_count_op(*array_nodata_list):
    """Count valid overlap.

    Args:
        array_nodata_list (list): a 2*n length list containing n arrays
            followed by n coresponding nodata values.

    Returns:
        a count of non-nodata overlaps for each element.

    """
    result = numpy.zeros(array_nodata_list[0].shape)
    n = len(array_nodata_list) // 2
    for array, nodata in zip(array_nodata_list[0:n], array_nodata_list[n::]):
        valid_mask = ~numpy.isclose(array, nodata)
        result[valid_mask] += 1
    return result


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


def main():
    """Entry point."""
    # convert raster list to just 1-10 integer
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, CLIPPED_DIR, OUTPUT_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)

    raster_path_list = [
        path for path in
        glob.glob(os.path.join(BASE_DATA_DIR, '*.tif'))
        if 'nodata0' in path]
    clipped_raster_path_list = [
        os.path.join(CLIPPED_DIR, os.path.basename(path))
        for path in raster_path_list]
    target_pixel_length = min([
        pygeoprocessing.get_raster_info(path)['pixel_size'][0]
        for path in raster_path_list])
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            raster_path_list, clipped_raster_path_list,
            ['bilinear'] * len(clipped_raster_path_list),
            [target_pixel_length, -target_pixel_length],
            # 'intersection'),
            [-124, 41, -121, 39]),
        hash_target_files=False,
        target_path_list=clipped_raster_path_list,
        task_name='clip align task')

    # overlap_raster_count_path = os.path.join(CHURN_DIR, 'overlap_count.tif')
    # task_graph.add_task(
    #     func=pygeoprocessing.raster_calculator,
    #     args=(
    #         raster_path_band_list + raster_nodata_list, overlap_count_op,
    #         overlap_raster_count_path, gdal.GDT_Byte, 0),
    #     target_path_list=[overlap_raster_count_path],
    #     task_name='count valid overlaps')

    # sum_task_list = []
    # align_task.join()
    # for raster_path in clipped_raster_path_list:
    #     sum_task = task_graph.add_task(
    #         func=sum_raster,
    #         args=((raster_path, 1),),
    #         dependent_task_list=[align_task],
    #         task_name='sum %s' % str(raster_path))
    #     sum_task_list.append(sum_task)

    # sum_list = [sum_task.get() for sum_task in sum_task_list]

    # target_sum_list = []
    # raster_path_band_list = []
    # raster_nodata_list = []
    # proportion_raster_band_path_list = []
    # prop_task_list = []
    # for path, sum_val in zip(clipped_raster_path_list, sum_list):
    #     if sum_val > 0:
    #         target_sum_list.append(0.5 * sum_val)
    #         raster_path_band_list.append((path, 1))
    #         raster_nodata_list.append(
    #             (pygeoprocessing.get_raster_info(path)['nodata'][0], 'raw'))
    #         proportional_path = os.path.join(
    #             PROPORTIONAL_DIR, f'prop_{os.path.basename(path)}')
    #         prop_task = task_graph.add_task(
    #             func=pygeoprocessing.raster_calculator,
    #             args=([
    #                 (path, 1), (sum_val, 'raw'),
    #                 (pygeoprocessing.get_raster_info(path)['nodata'][0],
    #                  'raw'),
    #                 (PROP_NODATA, 'raw')], proportion_op, proportional_path,
    #                 gdal.GDT_Float64, PROP_NODATA),
    #             target_path_list=[proportional_path],
    #             task_name=f'calculate proportion for {proportional_path}')
    #         proportion_raster_band_path_list.append((proportional_path, 1))
    #         prop_task_list.append(prop_task)

    # proportion_sum_raster_path = os.path.join(CHURN_DIR, 'prop_sum.tif')
    # task_graph.add_task(
    #     func=pygeoprocessing.raster_calculator,
    #     args=(
    #         [(PROP_NODATA, 'raw'), *proportion_raster_band_path_list],
    #         sum_rasters_op, proportion_sum_raster_path, gdal.GDT_Float64,
    #         PROP_NODATA),
    #     dependent_task_list=prop_task_list,
    #     target_path_list=[proportion_sum_raster_path],
    #     task_name='calc proportion sum')

    target_sum_list = []
    sum_task_list = []
    for raster_path in clipped_raster_path_list:
        sum_task_list.append(
            task_graph.add_task(
                func=sum_raster,
                args=((raster_path, 1),),
                dependent_task_list=[align_task],
                task_name='sum %s' % str(raster_path)))

    target_sum_list = [0.5 * sum_task.get() for sum_task in sum_task_list]

    task_graph.join()
    task_graph.close()
    del task_graph

    target_suffix = 'experimental'
    pygeoprocessing.raster_optimization(
        [(x, 1) for x in clipped_raster_path_list], target_sum_list,
        OUTPUT_DIR, target_suffix=target_suffix)

    return

    # target
    optimal_raw_mask_path = os.path.join(
        OUTPUT_DIR, f'optimal_mask_{target_suffix}.tif')

    smoothed_mask_path = os.path.join(
        OUTPUT_DIR, f'smoothed_mask_{target_suffix}.tif')

    smooth_radius = 3
    smooth_mask(optimal_raw_mask_path, smooth_radius, smoothed_mask_path)


def polygonize(base_raster_path_band, target_vector_path):
    """Polygonize base to target.

    Args:
        base_raster_path_band (str): path to base raster file.
        target_vector_path (str): path to vector that will be created making
            polygons over similar/connected regions.

    Returns:
        None.

    """
    raster = gdal.OpenEx(base_raster_path_band[0], gdal.OF_RASTER)
    band = raster.GetRasterBand(base_raster_path_band[1])

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(raster.GetProjection())

    driver = ogr.GetDriverByName('GPKG')
    poly_vector = driver.CreateDataSource(target_vector_path)

    poly_layer = poly_vector.CreateLayer(
        os.path.splitext(os.path.basename(target_vector_path))[0], spatial_ref,
        ogr.wkbPolygon)
    poly_layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    gdal.Polygonize(
        band, None, poly_layer, 0, ['CONNECTED8'],
        callback=gdal.TermProgress_nocb)


if __name__ == '__main__':
    main()
