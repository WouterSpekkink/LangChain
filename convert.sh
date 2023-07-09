#!/bin/bash

# One dir to keep papers that I have already processed
# and another dir to store newly added papers
existing_dir="/home/wouter/Documents/LangChain/data/old"
output_dir="/home/wouter/Documents/LangChain/data/new"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)

find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if [ -f "$existing_dir/$base_name.txt" ]; then
	echo "Text file for $file already exists, skipping."
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"
	counter=$((counter + 1))
	echo "Processed $counter out of $total PDFs."
    fi
done
