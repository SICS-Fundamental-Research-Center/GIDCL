## Instruction for pipeline training
### File Structure
Currently, we support data cleaning over these datasets:
```
Hospital
Beers
Rayyan
Tax
IMDB
Flights
Facilities
Inpatients
```
and they are valid for args `dataset_name`.

There original file is stored in folder `GEIL_Data/hospital/original`, while `clean.csv` is the clean version, `dirty.csv` is the dirty version.

To correctly load and evaluate, we do certain pre-processing on the above files, please check `utils/load_dataset.py` for details. Modify it if needed.

### Output

Currently, we put all the temp file and results in `output` folder. 

We upload dataset `Hospital` and its results. Below we illustrate how to run the pipeline, and take dataset `Hospital` as an example. 

### Graph Structure Learning(GSL)

run the command
```
python GSL_pipeline.py --dataset_name Hospital --embedding_model_path ../sentence_transformers/bge-small-en-1.5/
```
- Currently, we support `bge-small-en-1.5` and `bge-large-en-1.5` as the embedding model. Since we use FlagEmbedding to generate initial embedding, all cross-encoder model are supported. Just replace `--embedding_model_path` to your own path.
### Key Parameter
in `GSL_pipeline.py`, we list critical parameters as below:
- `generate_GSL_file`: whether to generate the GSL file. If set to `True`, the GSL file will be generated and stored in `PyG_Dataset/Hospital/raw/Hospital`. Otherwise, the GSL file will be loaded from the same place.
- `remove_GSL_cache`: whether to remove the GSL file after loading. If set to `True`, the GSL file will be removed after loading. Otherwise, the GSL file will be kept. Cache is typically saved in `PyG_Dataset/Hospital/processed` in `.pt` format.
- `add_semantic_embedding`: if set to `True`, the semantic embedding will be added to the node feature. Otherwise, the semantic embedding will not be added.
- `cluster_by_attr`: if set to `True`, it will initialize the clusters by the attributes that are most close to `cluster_divison_number`. Such setting is useful if `dirty_table` still have master data column, which can help better clustering, e.g. `ProviderNumber`. If set to `False`, the cluster division will be done by the `cluster_division_number` and randomly initialized.
- `cluster_division_number`: cluster number.
- `select_number`: selected example to annotated per cluster
### Output
- `GSL_pipeline.py` will generate `index.npy` as the selected labeled tuple; `clusters.npy` as the clustering result. 
- If you already selected the annotation result, just stored it in `output/Hospital/GSL`, in the format of `List` or `np.ndarray`. Then you can skip `GSL_pipeline.py` and come to detector training
- in `PyG_Dataset/Hospital/raw/Hospital`, `entity_df.csv` lists all the cells and attributes, which are deduplicated and listed as nodes; in `triple_df.csv`, the 1st column are all tuples; the 2nd column encodes different attributes as edges with different type(hetero graph). The 3rd are all nodes that are linked with the tuple.
- in `GSL`, we conduct link prediction task, with GCN over hetero graph as a self-supervised learning. The code is slightly modified from https://github.com/pyg-team/pytorch_geometric/blob/0d013cf488a722d5a5b3bf657302fa7ca8b6d120/examples/hetero/hetero_link_pred.py. Feel free to modify the loss function or the training code in `utils/graph_train.py`
### Detecor Training
run the command
```
python detector_training_pipeline.py --dataset_name Hospital
```
the previous `index.npy` and `clusters.npy` will be used to train the detector.
#### Rule Generation

First, you need to set an `vllm` server for LLM-based rule generation. e.g. you choose `Qwen-2.5-Coder-7B` model to generate rules, run the following code to start the vllm server:
```
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /home/user/model/Qwen2.5-Coder-7B --api-key token-example 
```
- in `detector_training_pipeline.py`, set `base_url` to your vllm server url, e.g. `http://localhost:8000` for local server
- Since all experiment are conducted on an offline server, we have not tried online model with this pipeline script. Feel free to replace `client` to online model api.
#### Key parameters
- `detector_model_path`: the path for the base detector model. Currently we use `ditto` with `roberta`. If you fail to install `apex`, later we will update a customized detector with `deberta`.
- `generate_detector_from_scratch`: if set to `True`, the pipeline will init `function_list` to empty, and generate rule from scratch. Otherwise, the pipeline will load `function_list.npy`.
- `generate_generator_from_scratch`: if set to `True`, the pipeline will use previous `function_list` to generate generation rules. Otherwise, the pipeline will load `function_list.npy`.
- `save_detector_training_result`: if set to `True`, the pipeline will save the detector training result 
- `save_pseudo_label_result`: if set to `True`, the pipeline will save the pseudo-labeled data, for further construction of the correction model training. 
### output
- `function_list.npy`: the generated function list(optional)
- `detector_training_result`: the training result of the detector(necessary)
- `pseudo_label_result`: the pseudo-labeled data for correction model training(necessary)
### Correction model training
run command
```
python correction_training_pipeline.py --dataset_name Hospital --base_model qwen
```
for training and inference

### Modified LLAMA-Factory
- We use llama-factory to conduct SFT stage per expert. You need to modify `src/llamafactory/data/loader.py` as below:
```
from datasets import DatasetDict, load_dataset, load_from_disk, Dataset
import pandas as pd
import numpy as np
```
- for function `def _load_single_dataset`, please insert the following code block after the final `else`:
```
    else:
        print(data_path,data_name,data_dir,data_files,dataset_attr.split,model_args.cache_dir,model_args.trust_remote_code)
        if data_args.tokenized_path is not None:
            dataset = load_from_disk(data_args.tokenized_path)
            print('load pre-defined arrow')
        elif data_args.train_file_path is not None:
            data_files = data_args.train_file_path.split(',')
            df = pd.DataFrame()
            for data_file in data_files:   
                df_current = pd.read_json(data_file)
                df_current['ids'] = df_current.index
                df = pd.concat([df,df_current])
            dataset = Dataset.from_pandas(df)
            print('loading from json file')
        else:           
            try:
                df = pd.DataFrame()
                for data_file in data_files:   
                    df_current = pd.read_json(data_file)
                    df = pd.concat([df,df_current])
                dataset = Dataset.from_pandas(df)
                print('loading from pandas')
            except:
                dataset = load_dataset(
                    path=data_path,
                    name=data_name,
                    data_dir=data_dir,
                    data_files=data_files,
                    split=dataset_attr.split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.hf_hub_token,
                    streaming=data_args.streaming,
                    num_proc=data_args.preprocessing_num_workers,
                    trust_remote_code=model_args.trust_remote_code,
                )
        print('load_dataset finished')

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)
```
This can skip huggingface-dataset check, and directly load the `json` file locally

- also modify `src/llamafactory/hparams/data_args.py` with the following args:
```
    train_file_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "json path for training file. "
            )
        },
    )
```

### script template
- please check `script` folder, here we list two training script for SFT.
- If you wan to replace a different model, please copy and modify the `yaml` file as template.
- You also need to modify
```
if base_model.lower().__contains__('qwen'):
    yaml_template = 'script/qwen2.5-7B-hospital-template.yaml'
elif base_model.lower().__contains__('mistral'):
    yaml_template = 'script/mistral-7B-hospital-template.yaml'
```
to your own template yaml file.

### SFT file generation
- cluster information is required for better example retrieval. It is optional, so feel free to ignore it.
- Pseudo-label generation result is required to avoid overfitting. It is optional, so feel free to ignore it.

### Training
- the python file will automatically use `llamafactory-cli train` to train the lora in `lora` folder, and use `vllm` to inference results with `vllm_query_qwen.py`. 
- If you want to change the default setting, please modify about 200-202 line in `correction_training_pipeline.py`

### Output
- the LLM generation is stored in `infrence` folder
- the corresponding correction result is stored in `output/Hospial/correction/correction.csv`. If you do not need to detect and fix VAD error, you can directly evaluate on `correction.csv`.

## FDs

run command
```
python FDs_update.py --dataset_name Hospital
```

### Required file
- `output/Hospial/correction/correction.csv` from previos step
- detection result for coreset selection.

### Key Parameters
- `alpha`: the `lambda` in paper. Used to control the weight of the tuples not in coreset.
- `correlation_thres`: filter false FDs in `correction.csv`. Default is 0.99.
- `one_to_one_variable_FDs`: if set to `True`, only detect possible `A -> B` FDs(necessary for `Hospital/Flights/IMDB`); if set to `False`, also detect `(A,B) -> C` FDs(necessary for dataset `Tax`).
- `skip_index`: if set to `True`, skip the index column in FD detection.

### Evaluation
- evalute the final result in precison/recall/F1, which definition is same with `Baran`
- Here we only list evaluation for `Hostpial`. More evaluation can be found in `eval.ipynb`

### Online Rule Generation
- Since most results are conducted on an offline server, we manually query and filter the online model `GPT-4`, and all the rules are listed in `function_set.ipynb` 