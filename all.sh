#!/bin/bash
set -e
for i in /mnt/picoshare/Ablage/Strahlprofile/Strahlprofile_neu/*/*/* ; do
	./beamprofile -i "$i" -g verlauf.png -m 500
done

