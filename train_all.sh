#!/bin/bash


python main.py --dataset_name cifar --gpu_used 0 --task_num 10 --file_name fetch_hyper_para --lamda_for_featureformer 350000 --lamda_for_featurelower 0 --lamda_for_pool3 350000 --lamda_for_prompt 400 --fc_inittype type8 --task_id 0 &


python main.py --dataset_name imagenet_a --gpu_used 1 --task_num 20 --file_name fetch_hyper_para --lamda_for_featureformer 70000 --lamda_for_featurelower 0 --lamda_for_pool3 70000 --lamda_for_prompt 500 --fc_inittype type8 --task_id 0 &

