# OnlineDistillGCN
Official code of [Online Adversarial Knowledge Distillation for Graph Neural Networks](https://arxiv.org/abs/2112.13966).

## Train
**For Citation datasets (Cora,Citeceer and Pubmed), run following command for example:**

``` bash
python train_citeceer_okd.py --arch GCN --data cora --warmup_epoch 100 --okd_epoch 100 --logitkd_trade 1 --gan_trade 1 --d_trade 1
```

**For PPI dataset, run following command for example:**

``` bash
python train_ppi_okd.py --arch GAT --warmup_epoch 50 --okd_epoch 50 --logit_kd_trade 1 --gan_trade 0.1 --d_trade 0.1 --num_hidden 4 --d_dim 64 --alpha 1
```

## Citation
```bibtex
@article{wang4261641online,
  title={Online Adversarial Knowledge Distillation for Graph Neural Networks},
  author={Wang, Can and Wang, Zhe and Chen, Defang and Zhou, Sheng and Feng, Yan and Chen, Chun},
  journal={Arxiv}
}
```


