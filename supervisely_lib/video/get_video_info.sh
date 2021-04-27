#!/bin/sh
file_path=$1
cpu_count=$2
ffprobe -i "$file_path" -threads $cpu_count -fflags +genpts -v error -print_format json -show_format -show_streams -show_frames -show_entries frame=stream_index,pkt_pts_time