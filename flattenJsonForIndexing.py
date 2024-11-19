import json
from datetime import datetime
from multiprocessing import Pool

def flatten_json(y):
    """
    Flatten a nested JSON object into a flat dictionary.
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def format_dates(flat_data):
    """
    Format date fields to Elasticsearch compatible format.
    """
    date_fields = [
        "DateCompleted_Year_text", "DateCompleted_Month_text", "DateCompleted_Day_text",
        "DateRevised_Year_text", "DateRevised_Month_text", "DateRevised_Day_text",
        "Article_Journal_JournalIssue_PubDate_Year_text", "Article_Journal_JournalIssue_PubDate_Month_text",
        "PubmedData_History_PubMedPubDate_Year_text", "PubmedData_History_PubMedPubDate_Month_text",
        "PubmedData_History_PubMedPubDate_Day_text"
    ]

    for field in date_fields:
        if field in flat_data:
            year = flat_data.get(f"{field.split('_text')[0]}_Year_text")
            month = flat_data.get(f"{field.split('_text')[0]}_Month_text")
            day = flat_data.get(f"{field.split('_text')[0]}_Day_text")

            if all([year, month, day]):
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                try:
                    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
                    flat_data[field.split('_text')[0]] = formatted_date
                    del flat_data[field]
                except ValueError:
                    print(f"Failed to parse date: {date_str}")

    # Handle month-only dates
    month_only_fields = [
        "Article_Journal_JournalIssue_PubDate_Month_text"
    ]

    for field in month_only_fields:
        if field in flat_data:
            year = flat_data.get(f"{field.split('_text')[0]}_Year_text")
            month = flat_data.get(f"{field.split('_text')[0]}_Month_text")

            if all([year, month]):
                date_str = f"{year}-{month.zfill(2)}-01"
                try:
                    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
                    flat_data[field.split('_text')[0]] = formatted_date
                    del flat_data[field]
                except ValueError:
                    print(f"Failed to parse date: {date_str}")

    return flat_data

def process_line(line):
    data = json.loads(line)
    
    # Flatten the JSON
    flat_data = flatten_json(data['MedlineCitation'])

    # Add PubmedData fields
    pubmed_data_flat = flatten_json(data['PubmedData'])
    flat_data.update(pubmed_data_flat)

    # Format dates
    formatted_data = format_dates(flat_data)
    
    return formatted_data

def transform_json(input_file_path, output_file_path=None):
    """
    Transform the JSON file from nested to flat structure and format dates.
    If output_file_path is provided, write the flattened data to a new file.
    Otherwise, return the flattened data.
    """
    with gzip.open(input_file_path, 'rt') as gz:
        
        with Pool(24) as pool:
            results = pool.map(process_line, gz.readlines())
        
        if output_file_path:
            with gzip.open(output_file_path, 'at') as f:
                for result in results:
                    json.dump(result, f)
                    f.write('\n')
        else:
            return results

# Example usage
import os
import gzip

input_json_path = "/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/compressedJson/"
output_json_path = "/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/compressedAndFlattenedJson/"

if not os.path.exists(output_json_path):
    os.makedirs(output_json_path)

from tqdm import tqdm

for file in tqdm(os.listdir(input_json_path)):
    if file.endswith(".gz"):
        input_file = os.path.join(input_json_path, file)
        output_file = os.path.join(output_json_path, f"{os.path.splitext(file)[0]}.gz")
        
        transform_json(input_file, output_file)

