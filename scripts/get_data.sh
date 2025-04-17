#!/bin/bash

# Define data locations
DATA_DIR="data/"

# Create directory for downloads
mkdir -p ${DATA_DIR}

# Download SwissProt
echo "Downloading SwissProt dataset..."
wget -P ${DATA_DIR} https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
echo "Extracting SwissProt..."
gunzip ${DATA_DIR}/uniprot_sprot.fasta.gz

echo "Download complete!"
