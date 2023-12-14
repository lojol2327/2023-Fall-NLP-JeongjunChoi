# 2023-Fall-NLP-Proejct T2M-Diff
Pytorch implementation of 2023-Fall-NLP-Project "T2M-Diff: Text-to-motion generation with Residual Quantization and Stochastic Refinement"


## 1. Installation

### 1.1. Environment

```bash
conda env create -f environment.yml
conda activate T2M-Diff
```

The code was tested on Python 3.8 and PyTorch 1.8.1.


### 1.2. Dependencies

```bash
bash dataset/prepare/download_glove.sh
```


### 1.3. Datasets


We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).   

Take HumanML3D for an example, the file directory should look like this:  
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```


### 1.4. Motion & text feature extractors:

We use the same extractors provided by [t2m](https://github.com/EricGuo5513/text-to-motion) to evaluate our generated motions. Please download the extractors.

```bash
bash dataset/prepare/download_extractor.sh
```

### 1.5. Pre-trained models 

The pretrained model files will be stored in the 'pretrained' folder:
```bash
bash dataset/prepare/download_model.sh
```


### 1.6. Render SMPL mesh (optional)

If you want to render the generated motion, you need to install:

```bash
sudo sh dataset/prepare/download_smpl.sh
conda install -c menpo osmesa
conda install h5py
conda install -c conda-forge shapely pyrender trimesh mapbox_earcut
```

## 2. Train

Note that, for kit dataset, just need to set '--dataname kit'.

### 2.1. VQ-VAE 

The results are saved in the folder output.

<details>
<summary>
VQ training
</summary>

```bash
python3 train_RVQ.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--num-quantizers 4 \
--dilation-growth-rate 3 \
--out-dir output/RVQ2 \
--dataname t2m \
--vq-act relu \
--quantizer residual \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name HumanML3D'
```

</details>

### 2.2. GPT 

The results are saved in the folder output.

<details>
<summary>
GPT and Diffusion model training
</summary>

```bash
python3 train_t2m_diff.py  \
--exp-name DIFF \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/RVQ2/HumanML3D/net_last.pth \
--vq-name RVQ2 \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer residual \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu
```

</details>

## 3. Evaluation 

### 3.1. VQ-VAE 
<details>
<summary>
VQ eval
</summary>

```bash
python3 VQ_eval.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--num-quantizers 4 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname t2m \
--vq-act relu \
--quantizer residual \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name TEST_RVQ2/HumanML3D \
--resume-pth output/RVQ2/HumanML3D/net_last.pth
```

</details>

### 3.2. GPT

<details>
<summary>
GPT-Diff eval
</summary>

Follow the evaluation setting of [text-to-motion](https://github.com/EricGuo5513/text-to-motion), we evaluate our model 20 times and report the average result. Due to the multimodality part where we should generate 30 motions from the same text, the evaluation takes a long time.

```bash
python3 GPT_eval_multi.py  \
--exp-name TEST_DIFF \
--batch-size 128 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/VQVAE/net_last.pth \
--vq-name RVQ2 \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer residual \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu \
--resume-trans output/GPT/net_best_fid.pth
```

</details>


## 4. SMPL Mesh Rendering 

<details>
<summary>
SMPL Mesh Rendering 
</summary>

You should input the npy folder address and the motion names. Here is an example:

```bash
python3 render_final.py --filedir output/TEST_DIFF/ --motion-list 000019 005485
```