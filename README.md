# dxda
## Paper
Please find the latest version of the paper [here](https://arxiv.org/abs/2001.02015)
## Dependencies
Tensorflow 1.x

python 3.4+

## Data
Please download MNIST-M data from the following [link](https://drive.google.com/open?id=15n7AgjnkIsxUtLxEA5YdKqxnIzNHCXed)

## Example on MNIST->MNIST-M
```bash
# Run Domain Alignment with Missing Classes in Target Training Set
# Example: when 7 classes are missing (3 left)
CUDA_VISIBLE_DEVICES=0 python3 baselines.py -missing_num 7
CUDA_VISIBLE_DEVICES=0 python3 unilateral.py -missing_num 7
```
## Results
Result in result.png


## Acknowledgement
Code is heavily borrowed from [tf-dann](https://github.com/pumpikano/tf-dann)

