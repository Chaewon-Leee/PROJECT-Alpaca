from pathlib import Path
import os
import json
import argparse

def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', '-f', default="data/python/train/", dest='file_path')
  parser.add_argument('--json_file_path', '-j', default="code_seed_tasks.jsonl", dest='json_file_path')

  file_path = parser.parse_args().file_path
  json_file_path = parser.parse_args().json_file_path

  return file_path, json_file_path

def code_to_json_format(file_path, json_file_path):
  Root_path = Path(os.getcwd())
  Root_path, alpaca_dir = os.path.split(Root_path)

  Data_Dir = os.path.join(Root_path, file_path)
  files = sorted(list(Path(Data_Dir).glob('*')))

  data = []
  for file in files:
    if not str(file)[-8:] == "original":
      with file.open() as f:
        data.append(f.readlines())

  answer_data, code_data, question_data = data
  question_data = [i.split("\n")[0] for i in question_data]
  answer_data = [i.split("\n")[0] for i in answer_data]
  code_data = [i.split("\n")[0] for i in code_data]

  dataset = []
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
    # json_format = {
    #   "instruction": question_data[idx],
    #     "input": code_data[idx],
    #     "output": answer,
    # }

    dataset.append(json_format)

  # file_path = os.path.join(Root_path, alpaca_dir, json_file_path)
  f = open(os.path.join(Root_path, alpaca_dir, json_file_path), 'w')
  json.dump(dataset, f)
  f.close()

  print(f"Done! The json file is saved in {os.path.join(Root_path, alpaca_dir, json_file_path)}")

if __name__ == "__main__":
  file_path, json_file_path = get_arguments()
  code_to_json_format(file_path, json_file_path)