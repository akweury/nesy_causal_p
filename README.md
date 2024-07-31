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

### Setup Remotely

build a docker

``` 
docker build -t ml-sha/arc_docker .
```
run the docker
``` 
docker run --gpus all -it -v /home/ml-jsha/storage/arc:/ARC/nesy_causal_p/storage --rm ml-sha/arc_docker

```

----
### Train

##### Train KP Perception Model
``` 
python -m src.train_perception --device 2
```