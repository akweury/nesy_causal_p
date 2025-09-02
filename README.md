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


### Docker
```
docker build -t grm:latest .
``` 

```
docker run -it --gpus all -v /home/ml-jsha/nesy_causal_p:/app --rm grm:latest
```
#### Train: GRM
``` 
python -m src.play
```
#### Train: Baseline Models


##### Llava-7B
``` 

```

##### InternVL-78B
``` 

```

#### Train: Ablation Study

```
python -m src.ablation_study --device 10 --task_id 6 --line_min_size 3
python -m src.ablation_study --device 0 --principle closure
python -m src.ablation_study --device 1 --principle similarity
python -m src.ablation_study --device 0 --principle proximity
python -m src.ablation_study --device 0 --principle continuity
python -m src.ablation_study --device 4 --principle symmetry

python -m mbg.scorer.train_context_aware_scorer --device 0 --n 200 --epochs 50 --principle proximity --sample_size 200 --data_num 100000
python -m mbg.scorer.train_context_aware_scorer --device 0 --n 120 --epochs 50 --principle symmetry --sample_size 200 --data_num 100000
python -m mbg.scorer.train_context_aware_scorer --device 1 --n 150 --epochs 50 --principle similarity --sample_size 50 --input_types color_size
python -m mbg.scorer.train_context_aware_scorer --device 1 --n 150 --epochs 50 --principle closure --sample_size 50
python -m mbg.object.train_patch_classifier --device 0 
``` 

---
#### Others
```
pip install -r requirements.txt
pip uninstall flash_attn flash_attn_2
```