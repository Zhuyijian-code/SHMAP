# Multimodal Hierarchical Fusion for Enhanced Long-Video Action Quality Assessment in Sports

Pytorch Implementation of paper:Multimodal Hierarchical Fusion for Enhanced Long-Video Action Quality Assessment in Sports

## Datasets

Here are the instructions for obtaining the features and videos for the Rhythmic Gymnastics and Fis-V datasets used in our experiments:

### Download extracted features and pretrained models

The extracted features and pretrained models can be downloaded from [here](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcHlFX0xmM1BGbDJpdGxwSXZ5ak56Y1EwVGxzUEE%5FZT1iaVE0Ync&cid=76593CF7B7FC849C&id=76593CF7B7FC849C%21175337&parId=76593CF7B7FC849C%21175222&o=OneUp) and should be placed in the current directory.

```
./
├── data/
└── pretrained_models
```

### For Rhythmic Gymnastics videos:

- Download the videos from the [ACTION-NET](https://github.com/qinghuannn/ACTION-NET) repository.

### For Fis-V videos:

- Download the videos from the [MS_LSTM](https://github.com/chmxu/MS_LSTM) repository.

Please use the above public repositories to obtain the features and videos needed to reproduce our results. 

## Installation

To get started, you will need to first clone this project and then install the required dependencies.

### Environments

- GPU: NVIDIA RTX 3090
- CUDA: 11.8
- Python: 3.8
- PyTorch: 2.4.1+cu118

### Basic packages

Install the required packages:

```bash
pip install -r requirements.txt
```

This will install all the required packages listed in the `requirements.txt` file.


## Training from scratch

Using the following command to train the model:

```bash
python main.py --gpu {gpu_id} --action {action_type}
```

### Command Arguments

- `--gpu_id`: The GPU device ID.
- `--action_type`: Ball, Clubs, Hoop, Ribbon, TES, PCS.


## Note

This repository serves as a demonstration for SHMAP. By debugging, you can quickly grasp SHMAP's configuration and build methods. Once the journal accepts the article, we will make all the code publicly available. For further inquiries, please email Professor Zhang ([yxzhang@fjnu.edu.cn](mailto:yxzhang@fjnu.edu.cn)).


## Remind

This code is directly related to our manuscript currently submitted to The Visual Computer. We encourage readers to cite this relevant manuscript. 

