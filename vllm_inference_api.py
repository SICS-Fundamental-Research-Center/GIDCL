import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.metrics import f1_score
import os
import time
import argparse
import subprocess
def json_to_csv_filename(json_file_path):
    base_name = os.path.basename(json_file_path)
    file_name, _ = os.path.splitext(base_name)
    csv_file_name = file_name + ".csv"
    return csv_file_name
def add_suffix_to_filename(file_path, suffix):
    directory, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_{suffix}{ext}"
    new_file_path = os.path.join(directory, new_file_name)
    return new_file_path
def main():
    parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")
    
    parser.add_argument("-test_file", "--file", help="Specify the input file")
    parser.add_argument("-json", "--json", action="store_true", help="Enable verbose mode")
    parser.add_argument("--count", type=int, default=1, help="Specify the count")
    parser.add_argument("-checkpoint_dir", "--directory", type=str, help="Specify the directory path")
    args = parser.parse_args()
    
    # if args.verbose:
    #     print("Verbose mode enabled")
    
    # if args.file:
    #     print(f"Input file: {args.file}")
    
    # print(f"Count: {args.count}")
    device = "CUDA_VISIBLE_DEVICES=%s" % str(args.count)
    file_path_list = [args.file]
    output_path = 'lora_weight/vicuna_temp_%s' % str(int(time.time()))
    command = "%s python src/export_model.py \
    --model_name_or_path vicuna-13b-1.3 \
    --finetuning_type lora \
    --checkpoint_dir %s \
    --output_dir %s \
    --template vicuna" %  (device,args.directory,output_path)  # 这里替换为你想要执行的Bash命令
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    llm = LLM(model=output_path, tensor_parallel_size=1)
    if(args.json):
        for file_path in file_path_list:
            result = pd.read_json(file_path)
            text_list = result['instruction'].to_list()
            prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:" % str(a) for a in text_list] ## Vicuna
            sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=512,logprobs=1)
            outputs = llm.generate(prompts, sampling_params)
            generation_list = []
            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                generation_list.append(generated_text)
            result['predict'] = generation_list
            print(result['predict'].value_counts())
            def Transfer(row):
                if(row['output']=='dismatch'):
                    label = 0
                else:
                    label = 1
                if(row['predict']=='dismatch'):
                    predict = 0
                else:
                    predict = 1
                return label,predict
            result_output = result.apply(Transfer,axis=1,result_type='expand')
            from sklearn.metrics import f1_score
            print(file_path)
            print(f1_score(y_true=result_output[0],y_pred=result_output[1]))
            result.to_csv('inference/vicuna-13b-%s' % json_to_csv_filename(args.file))
    else:
        text_list = np.load(args.file)
        prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:" % str(a) for a in text_list] ## Vicuna
        sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=16,logprobs=1)
        outputs = llm.generate(prompts, sampling_params)
        generation_list = []
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            generation_list.append(generated_text)
        print(add_suffix_to_filename(args.file, "output"))
        np.save(add_suffix_to_filename(args.file, "output"),np.array(generation_list))
    command_del = "rm -rf %s" % output_path
    subprocess.run(command_del, shell=True, capture_output=True, text=True)
if __name__ == "__main__":
    main()
