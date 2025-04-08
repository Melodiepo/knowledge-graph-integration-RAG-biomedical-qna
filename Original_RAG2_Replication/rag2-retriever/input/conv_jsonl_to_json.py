import json
import os

def conv_jsonl_to_json(input_jsonl_file, output_json_folder):
    # ensure the output folder exists
    os.makedirs(output_json_folder, exist_ok=True)

    # output json filename
    base_name = os.path.splitext(os.path.basename(input_jsonl_file))[0]
    output_json_file = os.path.join(output_json_folder, base_name + '.json')

    # read jsonl file and aggregate the data
    data = []
    with open(input_jsonl_file, 'r') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line_number}: {e}")
                continue

    # write to the json file
    with open(output_json_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Converted {input_jsonl_file} to {output_json_file}")


#doing this for the medqa data files            
input_jsonl_files = ["medqa/dev.jsonl", "medqa/test.jsonl", "medqa/train.jsonl"]
output_json_folder = "medqa/med_qa_json"

for jsonl_file in input_jsonl_files:
    conv_jsonl_to_json(jsonl_file, output_json_folder)

