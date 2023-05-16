import argparse
import json
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='mbart', choices=['mbart', 'indicbart'], help='model type')
parser.add_argument("--data_file", type=str, default="data.jsonl", help="data path")
parser.add_argument("--lang_pair", type=str, default="all", help="working language pair")
parser.add_argument("--output_dir", type=str, default=".", help="specify output data save path")
args = parser.parse_args()


#mBART supported languages
langs_mbart={"bn", "gu", "hi", "ml", "mni", "mr", "ta", "te", "ur", "en"}

#IndicBART supported languages
langs_indicbart={"as", "bn", "gu", "hi", "kn", "ml", "mni", "mr", "or", "pa", "ta", "te", "en"}

# To prepare a single csv file of all langs combinations use the code as it is, otherwise to get a single lang csv file specify the lang_pair
if args.lang_pair == "all":
    lang_combos = [str(x)+"-"+str(y) for x in "langs_"+args.model_type for y in "langs_"+args.model_type]
else:
    lang_combos = [args.lang_pair]

# Mapping of language codes to their corresponding mBART language tags
lang_map_mbart={"bn":"bn_IN", "gu":"gu_IN", "hi":"hi_IN", "ml":"ml_IN", "mni":"mni_IN", "mr":"mr_IN", "ta":"ta_IN", "te":"te_IN","ur":"ur_PK","en":"en_XX"}

# Create a dictionary to store the extracted data
extracted_data = {"train": [], "valid": [], "test": []}

# Iterate through the selected language combinations
for lang_pair in lang_combos:
	print("Processing lang-pair: "+str(lang_pair))
	doc_lang, sum_lang = [lang.strip() for lang in lang_pair.split("-")]
	with open(args.data_file) as input_file:
	    for line in input_file:
	        item = json.loads(line)
	        doc_key = doc_lang + "_text"
	        sum_key = sum_lang + "_summary"
	        if doc_key in item and sum_key in item and item[doc_key].strip() != "" and item[sum_key].strip() != "":
	        	if args.model_type == "mbart":
	        		extracted_data[item["split"]].append((item[doc_key].strip(), item[sum_key].strip(), lang_map_mbart[doc_lang], lang_map_mbart[sum_lang]))
	        	else:
	        		extracted_data[item["split"]].append((item[doc_key].strip(), item[sum_key].strip(), "<2"+doc_lang+">", "<2"+sum_lang+">"))

# Iterate through the extracted data dictionary	            	
for split, ls in extracted_data.items():
    fields = ['text', 'summary', 'src_lang', 'trg_lang']
    with open(args.output_dir + "/" + split + ".csv", "w", encoding="utf-8") as file:
        dict_list = []
        for item in ls:
        	sample = {"text": item[0],"summary": item[1], "src_lang": item[2], "trg_lang": item[3]}
        	dict_list.append(sample)

        #Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=fields)

        #Write header row if file is empty
        if file.tell() ==0:
        	writer.writeheader()

        #Iterate through the list of dictionaries and write each row to the file
        for d in dict_list:
        	writer.writerow(d)
