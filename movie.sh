#!/bin/bash
#rm -f *.mp4

name="$1"
num="$2"
CFL="$3"

num=$(printf "%04d" "$num")

CFL=$(printf "%.3f" "$CFL")

cd "./water_cup_output_weno5_cc_${name}_${num}_${CFL}"

ffmpeg -framerate 20 -pattern_type glob -i "*.png" -c:v h264 -pix_fmt yuv420p ${name}_${num}_${CFL}.mp4
mv *.mp4 ..
