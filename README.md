# DDM-RLIP
> paper code pytorch
## Discrete Diffusion Models with Refined Language–Image Pre-trained Representations for Remote Sensing Image Captioning

- 包括Sydney、UCM、NUPU的image caption数据集训练方法
- 后续更新具体训练方法

```bash
dataset=Sydney
batch_size=128
learning_rate=1e-5
epoch=720
bed=50000
date=0311_finetune
GPU_num=4
which_GPU=CUDA_VISIBLE_DEVICES=0,1,2,3

tmux new-session -s train_${dataset} -n editor -d
tmux send-keys -t train_${dataset} "rm -f ../dataset/${dataset}_Caption/train_image_raw_captions_tokens.pkl" C-m
tmux send-keys -t train_${dataset} "script -f ./log/useRSclip_noword_bs${batch_size}_lr${learning_rate}_epoch${epoch}_data${date}.log" C-m
tmux send-keys -t train_${dataset} "conda activate lengEnv" C-m
tmux send-keys -t train_${dataset} "${which_GPU} python -m torch.distributed.launch --nproc_per_node ${GPU_num} train_${dataset}_useRSclip_noWordMap.py --bs=${batch_size} --lr=${learning_rate} --tag=useRSclip_noword_bs${batch_size}_lr${learning_rate}_epoch${epoch}_data${date} --epochs=${epoch}" C-m
sleep ${bed}
tmux send-keys -t train_${dataset} "exit" C-m
tmux kill-session -t train_${dataset}

sleep 5
```
## 如果使用DDM-RLIP，请引用以下文章
```
@article{leng2024discrete,
  title={Discrete diffusion models with Refined Language--Image Pre-trained representations for remote sensing image captioning},
  author={Leng, Guannan and Xiong, Yu-Jie and Qiu, Chunping and Guo, Congzhou},
  journal={Pattern Recognition Letters},
  year={2024},
  publisher={Elsevier}
}
```
