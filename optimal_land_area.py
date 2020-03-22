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

from osgeo import gdal
from osgeo import osr
import ecoshard
import numpy
import pygeoprocessing
import taskgraph

BASE_DATA_DIR = 'data'
WORKSPACE_DIR = 'workspace_dir'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
CLIPPED_DIR = os.path.join(CHURN_DIR, 'clipped')
SMOOTHED_DIR = os.path.join(CHURN_DIR, 'smoothed')
TARGET_NODATA = -1
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def make_exponential_decay_kernel_raster(expected_distance, kernel_filepath):
    """Create a raster-based exponential decay kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Parameters:
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


def main():
    """Entry point."""
    # convert raster list to just 1-10 integer
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, CLIPPED_DIR, SMOOTHED_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 8)
    exponential_kernel_path = os.path.join(CHURN_DIR, 'exponential_kernel.tif')
    exponential_kernel_task = task_graph.add_task(
        func=make_exponential_decay_kernel_raster,
        args=(6, exponential_kernel_path),
        target_path_list=[exponential_kernel_path],
        task_name='make exponential kernel')

    raster_path_list = glob.glob(os.path.join(BASE_DATA_DIR, '*.tif'))
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
            ['near'] * len(clipped_raster_path_list),
            [target_pixel_length, -target_pixel_length],
            [-124, 41, -121, 39]),
        hash_target_files=False,
        target_path_list=clipped_raster_path_list,
        task_name='clip align task')

    byte_path_list = []
    for raster_path in clipped_raster_path_list:
        smoothed_raster_path = os.path.join(
            SMOOTHED_DIR, os.path.basename(raster_path))

        convolve_2d_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (raster_path, 1), (exponential_kernel_path, 1),
                smoothed_raster_path),
            kwargs={
                'ignore_nodata': True,
                'working_dir': CHURN_DIR,
                'target_nodata': TARGET_NODATA},
            dependent_task_list=[align_task],
            target_path_list=[exponential_kernel_path],
            dependent_task_list=[exponential_kernel_task, align_task],
            task_name='smooth %s' % os.path.basename(raster_path))

        byte_path = os.path.join(
            CHURN_DIR, os.path.basename(smoothed_raster_path))
        byte_path_list.append(byte_path)
        make_byte_raster_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(smoothed_raster_path, 1),
                 (TARGET_NODATA, 'raw'),
                 (255, 'raw')], clamp_to_integer, byte_path,
                gdal.GDT_Byte, 255),
            hash_target_files=False,
            target_path_list=[byte_path],
            dependent_task_list=[convolve_2d_task],
            task_name='clamp %s' % byte_path)
        task_graph.add_task(
            func=ecoshard.build_overviews,
            args=(byte_path,),
            kwargs={'overview_type': 'external'},
            kwargs={'interpolation_method': 'bilinear'},
            dependent_task_list=[make_byte_raster_task],
            task_name='build overviews for %s' % byte_path)

    task_graph.join()

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()
