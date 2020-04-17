#!/bin/bash -i
pip install git+https://github.com/richpsharp/pygeoprocessing.git@feature/raster_optimization --upgrade
python optimal_land_area.py
