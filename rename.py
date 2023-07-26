import bibtexparser

bibtex_file_path = '/home/wouter/Tools/Zotero/bibtex/library.bib'
output_txt_file = './renamed.txt'

def append_txt_extension(citation_keys):
    return [key + '.txt' for key in citation_keys]

def extract_citation_keys(bibtex_file, output_file):
    with open(bibtex_file, 'r') as bibfile:
        bib_database = bibtexparser.load(bibfile)

    citation_keys = [entry['ID'] for entry in bib_database.entries]
    modified_citation_keys = append_txt_extension(citation_keys)

    with open(output_file, 'w') as f:
        f.write("\n".join(modified_citation_keys))

if __name__ == "__main__":
    input_bibtex_file = bibtex_file_path

    extract_citation_keys(input_bibtex_file, output_txt_file)
    print("Citation keys extracted and saved to 'renamed.txt'.")

