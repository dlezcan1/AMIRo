#!/bin/bash
rosrun image_view image_saver image:=mono_camera/image_rect_color _filename_format:=$1"mono_%04d.%s" _save_all_image:=false  __name:=mono_image_saver &
sleep 1

while : 
do
	
	read -n 1 -p "Press 'Ctrl-C' to exit the program, enter any input to save an image."
	sleep 1	
	echo `date "+%m-%d-%Y_%H-%M-%S.%N"`
	rosservice call /mono_image_saver/save
	./getFBGPeaks.py -N 100 -d $1 $2
	echo 
	sleep 2

done

./addTimeStamps.bash $1
#kill $!
