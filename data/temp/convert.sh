#!/bin/bash

filename=$(basename "$1" .pdf)

pdftoppm -jpeg "$1" "${filename}"

for i in *.jpg
do
    echo "Processing $i"
    tesseract "$i" "$(basename "$i" .jpg)"
done

cat *.txt > "${filename}.txt"
