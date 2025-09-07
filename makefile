# ===== Config =====
IMAGE      := grm:latest
CODE_DIR   := /home/ml-jsha/nesy_causal_p
DATA_DIR   := /home/ml-jsha/storage/GRM_Data/coco_2017/selected   # contains val2017/ and annotations/
OUT_DIR    := /home/ml-jsha/storage/GRM_output

# Pipeline controls
STEPS      ?= detect,graph,train,infer,groupnms,eval
MAX_IMAGES ?=               # e.g. 200 for quick test
GPU_ID     ?= 0             # default GPU id

.PHONY: build run-docker run-local detect docker-detect help

help:
	@echo "make build | run-docker | run-local | detect | docker-detect"
	@echo "Override: STEPS=..., MAX_IMAGES=200, GPU_ID=1"

# ---- Build image ----
build:
	docker build -t $(IMAGE) .

# ---- Run in Docker (GPU) ----
run-docker:
	docker run --gpus all --rm -it \
	  -v "$(CODE_DIR):/app" \
	  -v "$(DATA_DIR):/mnt/data" \
	  -v "$(OUT_DIR):/app/run" \
	  -e CONFIG_PROFILE=remote \
	  -e DATA_ROOT=/mnt/data \
	  -e COCO_IMAGES=/mnt/data/val2017 \
	  -e COCO_ANN=/mnt/data/annotations/instances_val2017.json \
	  -e WORK_DIR=/workspace/run \
	  -e DEVICE=cuda:$(GPU_ID) \
	  -e NUM_WORKERS=8 \
	  -e MAX_IMAGES="$(MAX_IMAGES)" \
	  $(IMAGE) \
	  python -m src.run_coco.py --steps $(STEPS)

# ---- Only detect in Docker (GPU) ----
docker-detect:
	$(MAKE) run-docker STEPS=detect

# ---- Run locally (macOS CPU) ----
run-local:
	CONFIG_PROFILE=local \
	DEVICE=cpu \
	DATA_ROOT=$(DATA_DIR) \
	COCO_IMAGES=$(DATA_DIR)/val2017 \
	COCO_ANN=$(DATA_DIR)/annotations/instances_val2017.json \
	WORK_DIR=$(PWD)/.work \
	NUM_WORKERS=0 \
	MAX_IMAGES="$(MAX_IMAGES)" \
	python run_coco.py --steps $(STEPS)

# ---- Only detect locally ----
detect:
	$(MAKE) run-local STEPS=detect