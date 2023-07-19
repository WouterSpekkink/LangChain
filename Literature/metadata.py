import os
import bibtexparser

# Define the directory where the text files are stored
text_files_dir = './data/old/'

# Define the BibTeX file path
bibtex_file_path = '/home/wouter/Tools/Zotero/bibtex/library.bib'

# Read the BibTeX file
with open(bibtex_file_path) as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

# Get a list of all text file names in the directory
text_file_names = os.listdir(text_files_dir)

# Initialize an empty list to store metadata of matching entries
matching_entries = []

# Go through each entry in the BibTeX file
for entry in bib_database.entries:
    # Check if the 'file' key exists in the entry
    if 'file' in entry:
        # Extract the file name from the 'file' field and remove the extension
        pdf_file_name = os.path.basename(entry['file']).replace('.pdf', '')
        
        # Check if there is a text file with the same name
        if f'{pdf_file_name}.txt' in text_file_names:
            # If a match is found, append the metadata to the list
            matching_entries.append(entry)

# Print the metadata of matching entries
for entry in matching_entries:
    print(entry)
