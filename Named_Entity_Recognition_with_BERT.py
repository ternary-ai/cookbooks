
#Load Huggingface data
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import LukeForEntityClassification
from transformers import pipeline
import json, random

#Data Augentation
from bs4 import BeautifulSoup
import urllib
from urllib.request import Request
from urllib.error import HTTPError, URLError
import socket

#INPUT file
input_file="ner_pre_dataset20230925.csv.json"
#OUTPUT file
output_file = 'ner_post_dataset20230925_large.json'

append_mode=False
sample_only=False

#Custom functions
def get_page(url):
    """Scrapes a URL and returns the HTML source.

    Args:
        url (string): Fully qualified URL of a page.

    Returns:
        soup (string): HTML source of scraped page.
    """
    #print(url)
    #return
    req = Request(f"{json.loads(url)}", headers={'User-Agent': 'Mozilla/5.0'})

    soup=None
    try:
        response = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
    except HTTPError as error:
        print('Data not retrieved because %s\nURL: %s', error, url)
        return False
    except URLError as error:
        if isinstance(error.reason, socket.timeout):
            print('socket timed out - URL %s', url)
            return False
        else:
            print('some other error happened %s ' % error)
            return False
    else:
        
        try:
            soup = BeautifulSoup(response,
                         'html.parser',
                         from_encoding=response.info().get_param('charset'))
        except:
            soup = BeautifulSoup(response,
                                 'html.parser')
            return False
        
        element = soup.find('body')

        text_content = element.get_text(' | ',strip=True)

    return text_content

def write_json(new_data, filename='data.json'):

    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        #json.dump(file_data, file, indent = 4)
        json.dump(file_data, file, indent = 4, default=str)

# Load the JSON file into a Python data structure
with open(input_file, 'r') as json_file:
    data = json.load(json_file)

# Print the loaded data
print(f" data has {len(data)} observations")
random_index = random.randint(0, len(data) - 1)

#Load Models    
#modvar="AhmedTaha012/finance-ner-v0.0.9-finetuned-ner"
#modvar="dslim/bert-base-NER"
model_name="dslim/bert-large-NER"
#Setup AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

#Setup pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

#Take a randome sample from the data set
full_data=[]
if sample_only:
    tdata=data[random_index:random_index+5]
else:
    tdata=data

#Now execute the loop given the dataset
for index, entry in enumerate(tdata):
    
    print(f" processing {index}/{len(data)} => {round((index/len(data)*100),5)}% complete \n")
    source_value_lower = entry['_source'].lower()

    if 'youtu' in source_value_lower or 'redd.it' in source_value_lower or 'reddit' in source_value_lower or 'twitter' in source_value_lower or 'finclout' in source_value_lower:
        #print(" **** processing social **** \n")
        _content= entry['_content']
    else:
        try:
            cTmp=get_page(entry['_url'])
            if cTmp:
                _content= cTmp
            else:
                _content= entry['_content']

        except Exception as e:
            print(f"***** BS4 failed for URL: **** => {entry['_url']} *****")
            _content= entry['_content']

        
    ner_results = nlp(entry['_title'])
    entry.update({"_tEntities":ner_results})

    ner_results = nlp(_content)
    entry.update({"_cEntities":ner_results})
    entry.update({"_expContent":_content})

    full_data.append(entry)

    if append_mode:
        write_json(entry,output_file)

    
# Write the array of dictionaries to a JSON file
if not append_mode:
    with open(output_file, 'w') as json_file:
        json.dump(full_data, json_file, default=str)
