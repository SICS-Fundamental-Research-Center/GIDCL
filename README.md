# Graph-Enhanced Interpretable Data Cleaning with Large Language Models
This repository contains the code for the paper "Graph-Enhanced Interpretable Data Cleaning with Large Language Models"

We provide the checkpoint for our method in each stage, as well as the training/inference code for error detection and error correction.

## Data Download
- Due to the large size of the data and checkpoint, we upload all the related checkpoint into a zip file in [link](https://drive.google.com/file/d/1-D_bPJVsN6sTkcNKWJ-9BA3Mci-JpBPa/view?usp=drive_link). Please download the zip file and unzip the GEIL_ckpt folder to GEIL_ckpt folder.
- For all the data, please unzip the GEIL_Data.zip in the root folder and use the data in GEIL_Data
## Requirement

Please check the requirements in DITTO(https://github.com/megagonlabs/ditto), LLaMa-Factory(https://github.com/hiyouga/LLaMA-Factory) and vllm(https://github.com/vllm-project/vllm) for requirements.


## Original data
The dirty/clean table in stored in 
```
GEIL_Data/dataset/dirty.csv
GEIL_Data/dataset/clean.csv
e.g. for hospital, the dirty table is stored in 
GEIL_Data/hospital/dirty.csv
``` 
Except to the million-level data IMDB, all the clean/dirty table is equivalent to Baran paper(https://github.com/BigDaMa/raha/).

For Hospital and Flights dataset, we also provide the vary_error_rate result in 
```
GEIL_Data/dataset/dirty_error-rate.csv
```
(e.g. 10% error rate for hospital is in 
```
GEIL_Data/hospital/dirty_10.csv
```
). The clean table is the same.

## Sampled tuple index
The index of sampled tuple is stored in GEIL_Data/dataset/index.npy.

## Error Detection training/inference result

We apply a data-augmented sequence classification model to detect the error. The converged creator-critic training data is stored in 
```
GEIL_Data/dataset/detector/train.csv
GEIL_Data/dataset/detector/test.csv
```
For training/inference, please check `detector_train.ipynb` for details. The classifier is based on roberta-base LM model, and please replace hp_simple.lm args as well as all the roberta-base model path in ditto/model.py to your roberta-base model path

The detector result is outputed in an 1-d array, and you should pre-process it to fit the shape of the dirty_table for indicator(e.g. for beers dataset, we omit the first 2 rows index/id)

For reproduce, we provide the inference result in 
```
GEIL_Data/dataset/detector/detector.npy
```
## LLM-Generated Function Set
LLM Regex Query Example for rayyan-article_pagination:
```
The input 

[['May-51', '51-5'], ['Jun-70', '70-6'], ['Jun-93', '93-6'], ['11-Sep', '3111-9'], ['Aug-91', '91-8'], ['Jul-72', '72-7'], ['Jul-71', '71-7'], ['Apr-41', '3541-4']]

are [clean,dirty] cell pairs from table rayyan column article_pagination, and ['1187-9', '', '283-4', '714-9 ST  - [Noninvasive prenatal diagnosis of trisomy 21, 18 and 13 using cell-free fetal DNA]-', '835-40', '185-91', '163-7', '43-49', '1158-78', '158-61', '317-325', '10213-10224', '711-5', 'S40', '1245-50', '919-21', '185-90', 'Cd004797', '257-8', '991-5', '1304-16', '512-9 ST  - A performance improvement process to tackle tachysystole-', '1530-9', '2496-502', '785-794'] are examples of all cells from this columns. Please conclude a general pattern for dirty and clean cells, and write a general function with regular expression to detect whether a given cell is dirty or not. Input and output are all string format.
```

The generated function set is displayed in `function_set.ipynb`, and it is used for data augmentation for generating training data for error detection and error correction.



## Error Correction training/inference result
The training/testing data for fine-tuning LLM as $\mathcal{M}_G$ is provided in
```
GEIL_Data/dataset/correction/train.json
GEIL_Data/dataset/correction/test.json
```
and the fine-tuned LoRA checkpoint is provided in
```
GEIL_ckpt/dataset-train
```
For Training, please use the following command:
```
WANDB_MODE=disabled accelerate launch src/train_bash.py     
--stage sft     
--model_name_or_path vicuna-13b-1.3     
--do_train     
--finetuning_type lora     
--dataset hospital-train     
--output_dir lora_weight/hospital-train 
--overwrite_output_dir     
--lr_scheduler_type cosine     
--num_train_epochs 15.0     
--gradient_accumulation_steps 8     
--per_device_eval_batch_size 8     
--fp16     
--template vicuna     
--lora_r 16     
--quantization_bit 8 
--logging_steps 5 
--plot_loss  
--lora_target q_proj,v_proj 
--save_steps 50
```
Please replace the `model_name_or_path` to your model path, and add your own training file by modifying the `data/dataset_info.json`.

For training, the minimal requirement is 1\*32G GPU memory(1\*V100 recommended) for 13B model, and 1\*24G GPU memory(1\*3090 recommended) for model <=7B, and multiple GPU can speed up the training. Our test is conducted on vicuna-13b-1.3 model(https://huggingface.co/lmsys/vicuna-13b-v1.3) with 4\*V100 GPU.

For inference, please use the following command:
```
CUDA_VISIBLE_DEVICES=0 python vllm_inference_api.py -checkpoint_dir checkpoint-dir -test_file GEIL_Data/dataset/correction/test.json --count 0 --json
```
In `vllm_inference_api.py`, please replace the `model_name_or_path` to your model path for vicuna-13b-1.3 model(https://huggingface.co/lmsys/vicuna-13b-v1.3), and `output_path` to your output path for the temp model. The output will be stored in `inference` folder.

For reproduce, we also provide the inference result for $\mathcal{M}_G$ in 
```
GEIL_Data/dataset/correction/test_output.csv
```
For IMDB, due to the large size of correction, we cut the result in 4 parts.
### Output and Evaluation
After $\mathcal{M}_G$ and VAD correction, the output file is processed and stored in 
```
GEIL_Data/dataset/correction/result/correction.csv
```
And the evaluation for error detection/correction is provided in `eval.ipynb`.


