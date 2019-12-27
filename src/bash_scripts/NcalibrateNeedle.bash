#!/bin/bash

outdir="$(date "+%m-%d-%y_%H-%M")"/

while getopts 'o:d:v' option; do
	case "$option" in 
		d)
			outdir="$OPTARG"$outdir
			shift 2
			;;
	esac
done

mkdir -p $outdir

(trap 'break' SIGINT;
./rundatacollection.bash $outdir $@
)

./addTimeStamps.bash "$outdir"
