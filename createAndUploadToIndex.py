import json
from elasticsearch import Elasticsearch

import os
import gzip
from tqdm import tqdm




# Load secrets from a JSON file. Ensure that this file contains your Elasticsearch host and API key. 
secrets = json.load(open("secrets.json"))

# Ensure that the credentials in secrets.json are correct and have the necessary permissions.
client = Elasticsearch(secrets["elasticsearch"]["host"], 
                        api_key=secrets["elasticsearch"]["api_key"])

response = client.info()
print("Elasticsearch Info:", response)  # Check Elasticsearch connection and version info.


# Define the index name
index_name = "pubmed-articles"

mappings = {
    "properties": {
        "@Status": {"type": "keyword"},
        "@Owner": {"type": "keyword"},
        "PMID_@Version": {"type": "integer"},
        "PMID_#text": {"type": "keyword"},
        "DateCompleted_Year_#text": {"type": "date", "format": "yyyy"},
        "DateCompleted_Month_#text": {"type": "keyword"},
        "DateCompleted_Day_#text": {"type": "keyword"},
        "DateRevised_Year_#text": {"type": "date", "format": "yyyy"},
        "DateRevised_Month_#text": {"type": "keyword"},
        "DateRevised_Day_#text": {"type": "keyword"},
        "Article_@PubModel": {"type": "keyword"},
        "Article_Journal_ISSN_@IssnType": {"type": "keyword"},
        "Article_Journal_ISSN_#text": {"type": "keyword"},
        "Article_Journal_JournalIssue_@CitedMedium": {"type": "keyword"},
        "Article_Journal_JournalIssue_Volume_#text": {"type": "text"},
        "Article_Journal_JournalIssue_Issue_#text": {"type": "keyword"},
        "Article_Journal_JournalIssue_PubDate_Year_#text": {"type": "date", "format": "yyyy"},
        "Article_Journal_JournalIssue_PubDate_Month_#text": {"type": "keyword"},
        "Article_Journal_Title_#text": {"type": "text"},
        "Article_Journal_ISOAbbreviation_#text": {"type": "text"},
        "Article_ArticleTitle_#text": {"type": "text"},
        "Article_Pagination_MedlinePgn_#text": {"type": "text"},
        "Article_AuthorList_@CompleteYN": {"type": "keyword"},
        "Article_AuthorList_Author_@ValidYN": {"type": "keyword"},
        "Article_AuthorList_Author_LastName_#text": {"type": "text"},
        "Article_AuthorList_Author_ForeName_#text": {"type": "text"},
        "Article_AuthorList_Author_Initials_#text": {"type": "keyword"},
        "Article_Language_#text": {"type": "keyword"},
        "Article_GrantList_@CompleteYN": {"type": "keyword"},
        "Article_GrantList_Grant_GrantID_#text": {"type": "keyword"},
        "Article_GrantList_Grant_Agency_#text": {"type": "text"},
        "Article_GrantList_Grant_Country_#text": {"type": "text"},
        "Article_PublicationTypeList_PublicationType_@UI": {"type": "keyword"},
        "Article_PublicationTypeList_PublicationType_#text": {"type": "keyword"},
        "MedlineJournalInfo_Country_#text": {"type": "keyword"},
        "MedlineJournalInfo_MedlineTA_#text": {"type": "keyword"},
        "MedlineJournalInfo_NlmUniqueID_#text": {"type": "keyword"},
        "MedlineJournalInfo_ISSNLinking_#text": {"type": "keyword"},
        "ChemicalList_Chemical_RegistryNumber_#text": {"type": "keyword"},
        "ChemicalList_Chemical_NameOfSubstance_@UI": {"type": "keyword"},
        "ChemicalList_Chemical_NameOfSubstance_#text": {"type": "text"},
        "CitationSubset_#text": {"type": "keyword"},
        "MeshHeadingList_MeshHeading_DescriptorName_@UI": {"type": "keyword"},
        "MeshHeadingList_MeshHeading_DescriptorName_@MajorTopicYN": {"type": "text"},
        "MeshHeadingList_MeshHeading_DescriptorName_#text": {"type": "text"},
        "MeshHeadingList_MeshHeading_QualifierName_@UI": {"type": "keyword"},
        "MeshHeadingList_MeshHeading_QualifierName_@MajorTopicYN": {"type": "text"},
        "MeshHeadingList_MeshHeading_QualifierName_#text": {"type": "text"},
        "History_PubMedPubDate_@PubStatus": {"type": "keyword"},
        "History_PubMedPubDate_Year_#text": {"type": "date", "format": "yyyy"},
        "History_PubMedPubDate_Month_#text": {"type": "integer"},
        "History_PubMedPubDate_Day_#text": {"type": "integer"},
        "History_PubMedPubDate_Hour_#text": {"type": "integer"},
        "History_PubMedPubDate_Minute_#text": {"type": "integer"},
        "PublicationStatus_#text": {"type": "keyword"},
        "ArticleIdList_ArticleId_@IdType": {"type": "keyword"},
        "ArticleIdList_ArticleId_#text": {"type": "keyword"}
    }
}

# Check if the index exists and create it with mappings if not
if not client.indices.exists(index=index_name):
    client.indices.create(
        index=index_name,
        body={
            "mappings": mappings
        }
    )
    

# Directory containing the JSON files to be indexed
directory = "/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/compressedAndFlattenedJson"
from concurrent.futures import ThreadPoolExecutor

def index_file(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as json_file:
        for line in json_file:
            data = json.loads(line)
            client.index(index=index_name, body=data)

# Get list of files in the directory
files_to_index = [os.path.join(directory, filename) for filename in os.listdir(directory)]

# Parallelize indexing using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=32) as executor:  # You can adjust max_workers based on your system capabilities
    for _ in tqdm(executor.map(index_file, files_to_index), total=len(files_to_index)):
        pass