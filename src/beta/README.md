beta/
├── main.py                 # Entry point: orchestrates the pipeline
├── args_utils.py           # Argument parsing and configuration utilities
├── dataset/
│   ├── __init__.py
│   ├── loader.py           # load_tasks, load_dataset_tasks, etc.
│   └── image_utils.py      # load_image, load_annotation
├── features/
│   ├── __init__.py
│   ├── detector.py         # detect_features: extracts symbolic and neuro features from images
│   └── encoder.py          # encode_features: converts features into a fixed-shape object-centric matrix
├── reasoning/
│   ├── __init__.py
│   └── rules.py            # reasoning_rules: derives logic rules from encoded features
├── tokens/
│   ├── __init__.py
│   └── converter.py        # convert_rules_to_tokens: converts logical rules into natural language tokens via an LLM
└── utils/
    ├── __init__.py
    └── save_utils.py       # save_results and other utility functions