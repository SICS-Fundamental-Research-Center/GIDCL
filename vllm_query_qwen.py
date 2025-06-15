from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import sleep
import argparse
import json
from vllm.lora.request import LoRARequest
parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")

import json

def infer_type(value):
    if isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    else:
        return "null"

def generate_json_schema(data):
    schema = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {},
      "required": list(data.keys())
    }
    for key, value in data.items():
        property_schema = {"type": infer_type(value)}
        if isinstance(value, dict):
            property_schema = generate_json_schema(value)
        elif isinstance(value, list):
            if value:  # Non-empty list
                item_type = infer_type(value[0])
                property_schema = {"type": "array", "items": {"type": item_type}}
        schema["properties"][key] = property_schema
    return schema

if __name__ == "__main__":
    parser.add_argument("--llm_path", type=str, default='/data/home/wangys/model/Mistral-7B-Instruct-v0.2')

    # parser.add_argument("--data_name", type=str)

    parser.add_argument("--input_file", type=str)

    parser.add_argument("--lora_path", type=str, default='')

    parser.add_argument('--gpu_num',type=int,default=1)

    parser.add_argument('--gpu_memory_usage',type=float,default=0.9)

    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()





    llm_path = args.llm_path
    input_file = args.input_file
    gpu_num = args.gpu_num
    lora_path = args.lora_path
    gpu_mem = args.gpu_memory_usage
    output_path = args.output_path
    
    if(lora_path!=''):
        with open('{}/adapter_config.json'.format(lora_path).replace('//','/'), 'r', encoding='utf-8') as file:
            config = json.load(file)
        llm_path = config['base_model_name_or_path']
    ## Work with mistral/llama
    if(llm_path.lower().__contains__('mistral') or llm_path.lower().__contains__('llama')):
        llm = LLM(model=llm_path,
                tensor_parallel_size=gpu_num,
                enable_chunked_prefill=False,
                enforce_eager=True,
                num_scheduler_steps=10,  
                gpu_memory_utilization=gpu_mem,
                enable_lora=True,
                max_lora_rank=32
                #   enable_prefix_caching=False
                )  # Create an LLM.
    else:
        print(llm_path)
        if lora_path!='':
            llm = LLM(model=llm_path,
                    tensor_parallel_size=gpu_num,
                    #   enable_chunked_prefill=False,
                    enforce_eager=True,
                    enable_lora=True,
                    max_lora_rank=32
                    #   num_scheduler_steps=5,  
                    #   enable_prefix_caching=False
                    )  # Create an LLM.
        else:
            llm = LLM(model=llm_path,
                    tensor_parallel_size=gpu_num,
                    #   enable_chunked_prefill=False,
                    # enforce_eager=True,
                    # enable_lora=True
                    #   num_scheduler_steps=5,  
                    #   enable_prefix_caching=False
                    )  # Create an LLM.
    tokenizer = AutoTokenizer.from_pretrained(llm_path)


    try:
        text_list = pd.read_csv(input_file,index_col=0)['instruction'].to_list() ## the data_name args is a .csv file path, and text represents your query set
    except:
        result = pd.read_json(input_file)
        text_list = result['instruction'].to_list()

    prompts = [str(a) for a in text_list] ## Mistral
    text_all = []

    if(llm_path.lower().__contains__('mistral')): ## Mistral apply a unique prompt template!
        text_all = ["[INST] %s [/INST]" % str(a) for a in text_list]
    else:
        for prompt in tqdm(prompts):
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_all.append(text)



    sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=2048)
    if lora_path!='':
        outputs = llm.generate(text_all, 
                            sampling_params,
                            lora_request = LoRARequest('merge', 1, args.lora_path))
    else:
        outputs = llm.generate(text_all, 
                            sampling_params)
    generation_list = []
    # token_list_all = []
    # logprob_list_all = []
    # cumulative_logprob_all = []
    # decoded_token_list_all = []
    # Print the outputs.
    for output in outputs:
        logprob_lists = []
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # token_list = list(output.outputs[0].token_ids)
        # logprob_list = []
        # decoded_token_list = []
        # for element_dict in output.outputs[0].logprobs:
            
        #     key = list(element_dict.keys())[0]
            # logprob_value = element_dict[key].logprob
            # decoded_value = element_dict[key].decoded_token
            # logprob_list.append(logprob_value)
            # decoded_token_list.append(decoded_value)
        # cumulative_logprob_all.append(output.outputs[0].cumulative_logprob)
        # token_list_all.append(token_list)
        # logprob_list_all.append(logprob_list)
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        generation_list.append(generated_text)
        # decoded_token_list_all.append(decoded_token_list)
    # generation_stats = {}
    # generation_stats['text'] = generation_list
    # # generation_stats['token'] = token_list_all
    # # generation_stats['logprob'] = logprob_list_all
    # # generation_stats['cumulative_logprob'] = cumulative_logprob_all
    # # generation_stats['decoded_token'] = decoded_token_list_all
    # import numpy as np
    result['predict'] = generation_list
    # print(result['predict'].value_counts())
    result.to_csv(output_path)
    # def Transfer(row):
    #     if(row['output'].lower().__contains__('dismatch') or row['output'].lower().__contains__('mismatch')):
    #         label = 0
    #     else:
    #         label = 1
    #     if(row['predict'].lower().__contains__('dismatch') or row['output'].lower().__contains__('mismatch')):
    #         predict = 0
    #     else:
    #         predict = 1
    #     return label,predict
    # result_output = result.apply(Transfer,axis=1,result_type='expand')
    # from sklearn.metrics import f1_score,precision_score,recall_score
    # # print(file_path)
    # print(precision_score(y_true=result_output.iloc[:,0].to_list(),y_pred=result_output.iloc[:,1].to_list()),
    #       recall_score(y_true=result_output.iloc[:,0].to_list(),y_pred=result_output.iloc[:,1].to_list()),
    #       f1_score(y_true=result_output.iloc[:,0].to_list(),y_pred=result_output.iloc[:,1].to_list()))
    # np.save('enrich_data/enrich_query/{}_output_{}.npy'.format(data_name,llm_path.split('/')[-1]),np.array(generation_list))
    # np.save('{}.npy'.format(output_path),generation_stats)
    
    # text_list['predict'] = generation_list

