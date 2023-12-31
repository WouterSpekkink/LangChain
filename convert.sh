#!/bin/bash

# One file to keep the papers that I have already ingested
# One dir to store newly added papers
# A temporary dir for image-based pdfs.
existing_file="/home/wouter/Documents/LangChain_Projects/Literature/data/ingested.txt"
output_dir="/home/wouter/Documents/LangChain_Projects/Literature/data/new"
temp_dir="/home/wouter/Documents/LangChain_Projects/Literature/data/temp"

counter=0

total=$(find /home/wouter/Tools/Zotero/storage/ -type f -name "*.pdf" | wc -l)

find /home/wouter/Tools/Zotero/storage -type f -name "*.pdf" | while read -r file
do
    base_name=$(basename "$file" .pdf)

    if grep -Fxq "$base_name.txt" "$existing_file"; then
	echo -ne "Text file for $file already exists, skipping.\r"
    else 
	pdftotext -enc UTF-8 "$file" "$output_dir/$base_name.txt"

	pdfimages "$file" "$temp_dir/$base_name"
	
    fi
    counter=$((counter + 1))
    echo -ne "Processed $counter out of $total PDFs.\r"
    
done
