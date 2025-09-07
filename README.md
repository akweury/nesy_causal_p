# nesy_causal_p

The perception model based on neuro-symbolic and causal approaches

### Setup Locally

1. install pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. install requirements.txt

``` 
pip install -r requirements.txt
```

----
### Create Memory bank

##### Create Embedding
```  
python -m mbg.embedding_triangle --gpu 9
```

### Training Dataset:

/home/ml-jsha/storage/GRM_Data

### Clone Repository
```
git clone https://github.com/akweury/nesy_causal_p.git
```

### Docker
```
docker build -t grm:latest .
``` 

```
docker run -it --gpus all -v /home/ml-jsha/nesy_causal_p:/app -v /home/ml-jsha/storage/GRM_output/:/grm_output -v /home/ml-jsha/storage/GRM_Data/:/gen_data --rm grm:latest

``` 



#### Train: Baseline Models

##### ViT
``` 
python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model vit --device_id 6 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model vit --device_id 9 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model vit --device_id 10 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model vit --device_id 10 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model vit --device_id 5 --img_size 224 --remote

```

##### Llava-7B
``` 
python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model llava --device_id 6 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model llava --device_id 5 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model llava --device_id 4 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model llava --device_id 7 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model llava --device_id 5 --img_size 224 --remote
```

##### InternVL-78B
``` 
CUDA_VISIBLE_DEVICES=0,1,2 python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=0,1,2 python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=3,4,5 python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=4,5,7 python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=3,4,5 python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
```


#### Train: Ablation Study
```
python -m src.ablation_study --device 10 --task_id 6 --line_min_size 3
python -m src.ablation_study --device 0 --principle closure
python -m src.ablation_study --device 1 --principle similarity
python -m src.ablation_study --device 0 --principle proximity
python -m src.ablation_study --device 0 --principle continuity
python -m src.ablation_study --device 7 --principle symmetry --remote

python -m mbg.scorer.train_context_aware_scorer --device 0 --n 200 --epochs 50 --principle proximity --sample_size 200 --data_num 100000
python -m mbg.scorer.train_context_aware_scorer --device 4 --n 200 --epochs 10 --principle symmetry --sample_size 200 --data_num 100000 --remote
python -m mbg.scorer.train_context_aware_scorer --device 1 --n 150 --epochs 50 --principle similarity --sample_size 50 --input_types color_size
python -m mbg.scorer.train_context_aware_scorer --device 1 --n 150 --epochs 50 --principle closure --sample_size 50
python -m mbg.object.train_patch_classifier --device 0 
``` 


#### Analysis Results

``` 
python -m src.analysis_results --model principle --model vit --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model vit --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model vit --principle closure --img_num 3  
python -m src.analysis_results --model principle --model vit --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model vit --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model llava --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model llava --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model llava --principle closure --img_num 3  
python -m src.analysis_results --model principle --model llava --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model llava --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model internVL3-78B --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle closure --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model GRM --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle closure --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle continuity --img_num 3

```

---
#### Others
```
pip install -r requirements.txt
pip uninstall flash_attn flash_attn_2
```


---
## Coco Experiments

### Config

#### Local
``` 
export CONFIG_PROFILE=local DEVICE=cpu DATA_ROOT=/Users/jing/PycharmProjects/nesy_causal_p/storage/coco/selected WORK_DIR=$(pwd)/src/.work

python run_coco.py --steps detect
```

#### Remote

##### Run Detect
``` 
make docker-detect GPU_ID=0
make docker-graph GPU_ID=0

```