import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './InsuranceLM'
os.system('apt install git')
os.system('apt install git-lfs')
# please replace "your_git_token" with real token
os.system(f'git clone https://AntiSalt:dc991545bd84e9e6f6089f9a03c5a5ea6810d313@code.openxlab.org.cn/AntiSalt/InsuranceLM.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8B",
                description="""
InsuranceLM is a model that can help answer questions on insurances.  
                 """,
                 ).queue(1).launch()
