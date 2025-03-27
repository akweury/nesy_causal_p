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

##### Exp: Standard version

``` 
python -m src.play
```

docker run --gpus all -it -v /home/ml-jsha/storage/grm:/app/storage --rm grm 



### Docker
```

docker build -t grm .


python -m src.play --device 10


docker run --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ml-jsha/storage/grm/nesy_causal_p:/app -w /app  -it --rm grm 
```