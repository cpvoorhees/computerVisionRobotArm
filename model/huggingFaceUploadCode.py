from huggingface_hub import HfApi, Repository, upload_file
import os
from pathlib import Path

#hugging face account specifications
hf_username = ''                 #username for your hugging face account
repo_name = ''                   #repository name for where you want to your model
model_path = ''                  #local path to your model
token = ''                      #your token for you hugging face repository

repo_id = f"{hf_username}/{repo_name}"
p = Path(model_path)           #finds the file at the end of the path you set

print(f"Uploading {model_path} to {repo_id}")

#uploads the model into your hugging face repository
upload_file(
    path_or_fileobj=str(p),             #turns the windows path object into a file that can be read
    path_in_repo=p.name,                #finds the name of your file
    repo_id=repo_id,                   #finds the repo id
    repo_type="model",                 #letting hugging face know this is a model
    token=token                        #the token  of our repository
)