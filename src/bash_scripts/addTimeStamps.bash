#!/bin/bash

for f in $1*.jpg ;
do
	#echo $f
	newfile=$1monofbg_"$( date -r $f "+%m-%d-%Y_%H-%M-%S.%N.jpg" )"
	#echo $newfile
	cp -v $f $newfile
done
