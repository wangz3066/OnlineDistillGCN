# OnlineDistillGCN
Official code of [Online Adversarial Knowledge Distillation for Graph Neural Networks](https://www.sciencedirect.com/science/article/pii/S0957417423021735) (ESWA 2023).

## Train
**For Citation datasets (Cora,Citeceer and Pubmed), please run following command as example:**

``` bash
python train_citeceer_okd.py --arch GCN --data cora --warmup_epoch 100 --okd_epoch 100 --logitkd_trade 1 --gan_trade 1 --d_trade 1
```

**For PPI dataset, please run following command for example:**

``` bash
python train_ppi_okd.py --arch GAT --warmup_epoch 50 --okd_epoch 50 --logit_kd_trade 1 --gan_trade 0.1 --d_trade 0.1 --num_hidden 4 --d_dim 64 --alpha 1
```

## Citation
```bibtex
@article{wang2024online,
  title={Online adversarial knowledge distillation for graph neural networks},
  author={Wang, Can and Wang, Zhe and Chen, Defang and Zhou, Sheng and Feng, Yan and Chen, Chun},
  journal={Expert Systems with Applications},
  volume={237},
  pages={121671},
  year={2024},
  publisher={Elsevier}
}
```


