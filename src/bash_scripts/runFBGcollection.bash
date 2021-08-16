#!/bin/bash
mkdir -p $1
while : 
do
	
	read -n 1 -p "Press 'Ctrl-C' to exit the program, enter any input to save a frame of data."
	sleep 1	
	echo `date "+%m-%d-%Y_%H-%M-%S.%N"`
	./getFBGPeaks.py -N 200 -d $1 $2
	echo
	read -N 10000000 -t 0.05  
	sleep 2

done

#kill $!
