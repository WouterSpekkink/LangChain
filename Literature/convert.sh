#!/bin/bash

# One dir to keep papers that I have already processed
# and another dir to store newly added papers
existing_file="/home/wouter/Documents/LangChain/Literature/data/ingested.txt"
output_dir="/home/wouter/Documents/LangChain/Literature/data/new"
temp_dir="/home/wouter/Documents/LangChain/Literature/data/temp"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)

find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if grep -Fxq "$base_name.txt" "$existing_file"; then
	echo "Text file for $file already exists, skipping."
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"

	pdfimages "$file" "$temp_dir/$base_name"
	
	file_exists=false
        for image_file in "$temp_dir/$base_name-"*.pbm "$temp_dir/$base_name-"*.ppm; do
            if [ -e "$image_file" ]; then
                file_exists=true
                break
            fi
        done
        
        if  [ "$file_exists" = true ]
        then
            rm "$output_dir/$base_name.txt"
        fi
	
    fi
    counter=$((counter + 1))
    echo "Processed $counter out of $total PDFs."

done
