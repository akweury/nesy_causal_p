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

##### Exp: Standard version

``` 
python -m src.play
```

docker run --gpus all -it -v /Users/jing/PycharmProjects/nesy_causal_p:/app --rm grm 



### Docker
```


docker build -t grm:latest .

docker run -it --gpus all -v /home/ml-jsha/nesy_causal_p:/app --rm grm:latest
  
python3 -m debugpy --wait-for-client --listen 0.0.0.0:5678 play.py

python -m src.play --device 10 --task_id 6 --line_min_size 3
python -m src.play --device 10 --principle similarity
 
```



