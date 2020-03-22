"""Try to get minimum land with no less than 90% of service."""
import glob
import logging
import os
import sys

from osgeo import gdal
import ecoshard
import numpy
import pygeoprocessing
import taskgraph

BASE_DATA_DIR = 'data'
WORKSPACE_DIR = 'workspace_dir'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
CLIPPED_DIR = os.path.join(CHURN_DIR, 'clipped')

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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
    try:
        os.makedirs(CHURN_DIR)
    except OSError:
        pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 8)

    raster_path_list = glob.glob(os.path.join(BASE_DATA_DIR, '*.tif'))
    byte_path_list = []
    target_pixel_length = 100
    for raster_path in raster_path_list:
        raster_info = pygeoprocessing.get_raster_info(raster_path)
        target_pixel_length = min(
            target_pixel_length, raster_info['pixel_size'][0])
        LOGGER.debug(raster_info['raster_size'])
        byte_path = os.path.join(CHURN_DIR, os.path.basename(raster_path))
        byte_path_list.append(byte_path)
        make_byte_raster_task = task_graph.add_task(
            func=pygeoprocessing.raster_calculator,
            args=(
                [(raster_path, 1),
                 (raster_info['nodata'][0], 'raw'),
                 (255, 'raw')], clamp_to_integer, byte_path,
                gdal.GDT_Byte, 255),
            hash_target_files=False,
            target_path_list=[byte_path],
            task_name='clamp %s' % byte_path)
        task_graph.add_task(
            func=ecoshard.build_overviews,
            args=(byte_path,),
            target_path_list=[byte_path],
            hash_target_files=False,
            kwargs={'interpolation_method': 'bilinear'},
            dependent_task_list=[make_byte_raster_task],
            task_name='build overviews for %s' byte_path)

    clipped_raster_path_list = [
        os.path.join(CLIPPED_DIR, os.path.basename(path))
        for path in byte_path_list]
    task_graph.join()

    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            byte_path_list, clipped_raster_path_list,
            ['near'] * len(clipped_raster_path_list),
            [target_pixel_length, -target_pixel_length],
            [-124, 41, -121, 39]),
        hash_target_files=False,
        target_path_list=clipped_raster_path_list,
        task_name='align task')

    for clipped_path in clipped_raster_path_list:
        task_graph.add_task(
            func=ecoshard.build_overviews,
            args=(clipped_path,),
            target_path_list=[clipped_path],
            hash_target_files=False,
            kwargs={'interpolation_method': 'bilinear'},
            dependent_task_list=[align_task],
            task_name='build overviews for %s' % clipped_path)

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()
