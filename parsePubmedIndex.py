import json
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import pubmedIndexParser
from multiprocessing import Pool
    


import os
import gzip



def process_file(file_name):
        
    full_file_path = os.path.join(xml_file_path, file_name)
    with gzip.open(full_file_path, 'rt', encoding='utf-8') as f:
        
        # Assuming that the pubmedIndexParser.parse function takes an XML string and returns a list of documents
        documents = pubmedIndexParser.xml_to_json(f.read())
    
    return documents



secrets = json.load(open("secrets.json"))

# Ensure that the credentials in secrets.json are correct and have the necessary permissions.
client = Elasticsearch(secrets["elasticsearch"]["host"], 
                        api_key=secrets["elasticsearch"]["api_key"])

client.options()
response = client.info()
# print("Elasticsearch Info:", response)  # Check Elasticsearch connection and version info.


# Define the index name
index_name = "pubmed-articles"

# Check if the index exists and create it if not
if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name)
    
    



if __name__ == "__main__":
    
    
    xml_file_path = '/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    output_file_path = '/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/compressedJson/'

    file_list = [file_name for file_name in os.listdir(xml_file_path) if file_name.endswith('.xml.gz')]
    
    # Using multiprocessing to process files in parallel
    def write_documents_to_gzip(documents, output_file_name):
        with gzip.open(output_file_name, 'wt', encoding='utf-8') as f:
            for doc in documents:
                json.dump(doc, f)
                f.write('\n')

    def process_and_write(file_name):
        documents = process_file(file_name)
        output_file_name = os.path.join(output_file_path, file_name.replace('.xml.gz', '.json.gz'))
        write_documents_to_gzip(documents, output_file_name)

    with Pool(32) as pool:
        list(tqdm(pool.imap_unordered(process_and_write, file_list), total=len(file_list)))
        


    # # Indexing documents in Elasticsearch
    # actions = [
    #     {
    #         "_index": index_name,
    #         "_source": doc
    #     }
    #     for doc in documents 
    # ]
    # bulk(client, actions)