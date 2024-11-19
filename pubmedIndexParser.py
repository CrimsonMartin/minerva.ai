import xml.etree.ElementTree as ET
import json

def parse_element(element):
    parsed_dict = {}
    
    # Process attributes
    parsed_dict.update(('@' + k, v) for k, v in element.attrib.items())
    
    # Process child elements
    children = list(element)
    if children:
        parsed_dict.update({child.tag: parse_element(child) for child in children})
    else:
        # If no children, store the text content
        parsed_dict['#text'] = element.text.strip() if element.text else ''
    
    return parsed_dict

def xml_to_json(xml_content):
    # Parse the XML string into an ElementTree object
    root = ET.fromstring(xml_content)
    
    # Initialize a list to hold individual PubmedArticle dictionaries
    pubmed_articles_list = []
    
    # Iterate over each 'PubmedArticle' element in the XML content
    for article_element in root.findall('.//PubmedArticle'):
        # Convert the ElementTree object of PubmedArticle to a dictionary
        parsed_dict = parse_element(article_element)
    
        # Append the dictionary to the list of articles
        pubmed_articles_list.append(parsed_dict)
    
    # Convert the list of dictionaries to a JSON string
    return pubmed_articles_list

if __name__ == "__main__":
    xml_file_path = '/mnt/gonzalez/source/minerva.ai/ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed24n0001.xml'
    with open(xml_file_path, 'r') as file:
        contents = file.read()  # Read the XML file content.
    
    articles = xml_to_json(contents)
    
    for article in articles[:5]:  # Print first 5 articles
        print(json.dumps(article, indent=4))
