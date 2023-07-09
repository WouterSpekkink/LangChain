#!/bin/bash

output_dir="/home/wouter/Documents/LangChain/data"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)


find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if [ -f "$output_dir/$base_name.txt" ]; then
	echo "Text file for $file already exists, skipping."
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"
	counter=$((counter + 1))
	echo "Processes $counter out of $total PDFs."
    fi
done
