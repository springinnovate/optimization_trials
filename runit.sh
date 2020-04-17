#!/bin/bash -i
pushd /usr/local
git clone https://github.com/richpsharp/pygeoprocessing.git
pushd pygeoprocessing
git checkout feature/raster_optimization
python setup.py install
popd
popd
python optimal_land_area.py
