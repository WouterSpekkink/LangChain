# Introduction
This repo contains one of my current langchain projects.
This project focuses on augmented retrieval with academic publications in my Zotero library. 
This project makes use of several python scripts:
- A script to make a new index when starting from scratch (indexer.py)
- A script to update an existing index after new papers have been added to the library (updater.py)
- A script to remove corrupt documents from the index (remove_docs.py)
- A script that does the actual Q&A, using chainlit for the UI and GPT3.5-turbo-16k as the main LLM (ask_cl.py)

