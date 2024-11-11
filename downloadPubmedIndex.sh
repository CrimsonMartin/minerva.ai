wget -r -np -nH --cut-dirs=3 -R "index.html*" ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
gunzip *.gz
