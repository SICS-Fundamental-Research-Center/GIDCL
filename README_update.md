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
`detector_model_path`: the path for the base detector model. Currently we use `ditto` with `roberta`. If 