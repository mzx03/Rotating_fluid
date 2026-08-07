#!/bin/bash

name="$1"
ncell="$2"
CFL="$3"

python3 evolution_1D_WENO5.py $name $ncell $CFL

bash movie.sh $name $ncell $CFL
