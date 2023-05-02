from pathlib import Path
import os
import json

def code_to_json_format(data_path="data/python/train/", json_path="code_dataset.jsonl"):
  Root_path = Path(os.getcwd())
  Root_path, alpaca_dir = os.path.split(Root_path)
  Data_Dir = os.path.join(Root_path, data_path)
  files = list(Path(Data_Dir).glob('*'))

  data = []
  for file in files:
    if not str(file)[-8:] == "original":
      with file.open() as f:
        data.append(f.readlines())

  dataset = []
  answer_data, code_data, question_data = data
  question_data = [i.split("\n")[0] for i in question_data]
  answer_data = [i.split("\n")[0] for i in answer_data]
  code_data = [i.split("\n")[0] for i in code_data]

  for idx, answer in enumerate(answer_data):
    json_format = {
      "id": f"seed_task_{idx}",
      "name": f"name_{idx}",
      "instruction": question_data[idx],
      "instances": [{
        "input": code_data[idx],
        "output": answer,
      }],
      "is_classification": False
    }
    dataset.append(json_format)

  f = open(os.path.join(Root_path, alpaca_dir, json_path), 'w')
  json.dump(dataset, f)
  # for i in dataset:
    # json.dump(i, f)
  f.close()
