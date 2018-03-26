#!/bin/bash
set -e
for i in ~/Strahlprofile/*/*/* ; do
	./beamprofile -i "$i" -g verlauf.png -m 500
done

