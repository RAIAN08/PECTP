
# Class-Incremental Learning with Parameter-Efficient Cross-Task Prompts


## Introduction
Class-Incremental Learning (CIL) aims to learn deep models on sequential tasks continually, where each new task includes a batch of new classes and deep models do not have access to task-ID information at the inference time. Recent vast pre-trained models (PTMs) have achieved outstanding performance by prompt technique in practical CIL without the old samples (rehearsal-free) and with a memory constraint (memory-constrained): Prompt-extending and Prompt-fixed methods. However, prompt-extending methods need a large memory buffer to maintain an ever-expanding prompt pool and meet an extra challenging prompt selection problem. Prompt-fixed methods only learn a single set of prompts on one of the incremental tasks and can not handle all the incremental tasks effectively. To achieve a good balance between the memory cost and the performance on all the tasks, we propose a Parameter-Efficient Cross-Task Prompt (PECTP) framework with Prompt Retention module (PRM) and Classification Layer Retention module (LRM). To make the ﬁnal learned prompts effective on all incremental tasks, PRM constrains the evolution of cross-task prompts' parameters from Outer Prompt Granularity and Inner Prompt Granularity. Besides, we employ LRM to inherit old knowledge in the last classification layer to facilitate the cross-tasks prompts' generalization ability. Extensive experiments show the effectiveness of our method.





![image-20240118175522482](image-20240118175522482.png)



![image-20240118175424076](image-20240118175424076.png)




<h3>TL;DR</h3>


A Parameter-Efficient Cross-Task Prompts learning framework, with a fixed number of prompts dynamically learning on each incremental task. 
Extensive experiments on  seven benchmarks show that **PECTP** can beat SOTA with only a few tunable parameters.
</div>



## Requirements
### Environment
1. [torch 1.11.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.12.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)


### Dataset
We provide the processed datasets as follows:
- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**:  Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ImageNet-A**:Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
- **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
- **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) if the file is too large to download.

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./utils/data.py`  according to your own path.

## Running scripts
Please follow the settings in the `exps` folder to prepare your json files, and then run:

```
sh run_cifar.sh
```

or
```
python main.py --gpu_used 1 --task_num 10 --file_name fetch_hyper_para --lamda_for_featureformer 350000 --lamda_for_featurelower 0 --lamda_for_pool3 350000 --lamda_for_prompt 400 --fc_inittype type8 --task_id 0
```


## Acknolegment
This repo is based on  [PyCIL](https://github.com/G-U-N/PyCIL) and [ADAM](https://github.com/zhoudw-zdw/RevisitingCIL).



python main.py --gpu_used 2 --task_num 20 --file_name fetch_hyper_para --lamda_for_featureformer 70000 --lamda_for_featurelower 0 --lamda_for_pool3 70000 --lamda_for_prompt 500 --fc_inittype type8 --task_id 0
