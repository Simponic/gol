#!/bin/bash

if [ $# == 5 ]; then
  for x in $(find $1 -type f -name "*.bin"); do
    ./convert_bin_to_img $x $2 $3
  done
  ffmpeg -r $4 -f image2 -s "${2}x${3}" -i "${1}/iteration-%07d.bin.bmp" -vcodec libx264 -crf 25  -pix_fmt yuv420p "${5}.mp4"
else
  echo "Usage: ./make_movie.sh <directory> <width> <height> <fps> <output_video_name>"
fi

