# SEMICON: A Learning-to-hash Solution for Large-scale Fine-grained Image Retrieval
--------------------------
The hyper-parameter for \alpha in the paper is 0.15 (It's 0.3 in the paper and we have changed it in the code. This will result in about a 1~2-point fluctuation in mAP. If the replication error is 3 points or more, there must be an issue with the code or environment, so please check the code and environment carefully.). We have also provided some training logs for the CUB, NABirds, and Food101 datasets (Logs for ECCV 2022 version please cf SEMICON_log, we have not provide paper for SEMICON++ version). 

If you find significant discrepancies in the reproduced results, you can contact us and we will do our best to address your concerns (Please provide your log files in the email. Otherwise, we cannot determine where the problem lies.).

Paper Link: https://arxiv.org/pdf/2209.13833

## Environment

Python 3.8.5  
Pytorch 1.10.0  
torchvision 0.11.1  
numpy 1.19.2
loguru 0.5.3
tqdm 4.54.1

--------------------------
## Dataset
We use the following 5 datasets: CUB200-2011, Aircraft, VegFru, Food101 and NABirds.

--------------------------
## Train

We train our model in only one 2080Ti card, for different datasets, we provide different sample training commands:  

The CUB200-2011 dataset:

     python run.py --dataset cub-2011 --root /dataset/CUB2011/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'CUB-SEMICON' --momen=0.91

The Aircraft dataset:

     python run.py --dataset aircraft --root /dataset/aircraft/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'Aircraft-SEMICON' --momen=0.91

The VegFru dataset:

     python run.py --dataset vegfru --root /dataset/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'VegFru-SEMICON' --momen=0.91

The Food101 dataset:

     python run.py --dataset food101 --root /dataset/food101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info 'Food101-SEMICON' --momen 0.91

The NAbirds dataset:
     
     python run.py --dataset nabirds --root /dataset/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'NAbirds-SEMICON' --momen=0.91

--------------------------
## Test

Taking the CUB200-2011 dataset as an example, the testing command is:  

     python run.py --dataset cub-2011 --root /dataset/CUB2011/CUB_200_2011 --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'CUB-SEMICON'


