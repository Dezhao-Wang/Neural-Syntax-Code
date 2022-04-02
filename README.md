# Neural-Syntax

Code for CVPR 2022 Paper "NEURAL DATA-DEPENDENT TRANSFORM FOR LEARNED IMAGE COMPRESSION"

Author: Dezhao Wang, Wenhan Yang, Yueyu Hu, Jiaying Liu

Arxiv Link: <https://arxiv.org/abs/2203.04963>

Project Page: <https://dezhao-wang.github.io/NeuralSyntax-Website/>

## Evaluation

*   Here we provide our [pretrained model](https://drive.google.com/file/d/1Cp3foBl926vAvmWtk-vji_2-LPNom840/view?usp=sharing) optimized for MSE at lambda 0.0015 for reference.

```bash
python eval.py --data_path {test_set_path} --lambda {lambda_value} --weight_path {tested_checkpoint_path} [--tune_iter {pre_prpcessing_tune_iteration_num}] [--post_processing] [--pre_processing] [--high]
e.g. python eval.py --data_path ../Kodak/ --lambda 0.0015 --weight_path ./weights/mse0.0015.ckpt --post_processing --pre_processing
```

## Training

*   We use a two-stage training strategy.

    *   In the first stage, we train the transform model and entropy model.

    *   In the second stage, we train the post-processing model with other modules fixed.

```bash
python train.py --train_data_path {"train_set_path/*"} --lambda {lambda_value} --checkpoint_dir {saved_checkpoint_dir}  [--weight_path {pretrain_model}] [--batch_size {batch_size}] [--lr {learning rate}] [--post_processing] [--high]
e.g. python train.py --train_data_path "../DIV2K_train_HR/*png" --lambda 0.0015 --checkpoint_dir ./weights/  --weight_path ./weights/mse0.0015.ckpt
```

## Acknowledgement

We implement our post-processing model with the help of <https://github.com/wwlCape/HAN>.
