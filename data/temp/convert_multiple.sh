#!/bin/bash

output_dir="/home/wouter/Documents/LangChain/data/new/"
pbm_directory="/home/wouter/Documents/LangChain/data/temp"

# Create an associative array
declare -A base_names

# Handle filenames with spaces by changing the Internal Field Separator (IFS)
oldIFS="$IFS"
IFS=$'\n'

# Go through each file in the PBM directory
for file in "$pbm_directory"/*.pbm "$pbm_directory"/*.ppm
do
    # Get the base name from the path
    base_name=$(basename "$file" | rev | cut -d- -f2- | rev)

    # Add the base name to the associative array
    base_names["$base_name"]=1
done

# Restore the original IFS
IFS="$oldIFS"

# Go through each unique base name
for base_name in "${!base_names[@]}"
do
    # Remove any existing text file for this base name
    rm -f "$output_dir/$base_name.txt"

    # Go through each PBM file for this base name, handling spaces in filenames
    for ext in pbm ppm
    do
        find "$pbm_directory" -name "$base_name-*.$ext" -print0 | while read -r -d $'\0' file
        do
            # OCR the file and append the results to the text file
	    echo "Converting $file" 
            tesseract "$file" stdout >> "$output_dir/$base_name.txt"
        done
    done
done
