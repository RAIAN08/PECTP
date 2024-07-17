import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from torch.utils.data import DataLoader


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    INIT_TYPE = 'share' if args['intra_share'] == 1 else 'unshare'
    attn_map_file = {} # save attn result from each incremental learning step
    logger_list = [] # save logging.info...print at the end
    logger_list_new = []
    logger_list_old = []
    if 'in21k' in args['convnet_type']:
        backbone = 'in21k'
    else:
        backbone = 'in1k'

    logs_name = "logs/{}/{}/{}/{}/{}/{}/{}/".format(args['dataset'], backbone, args["model_name"], args["dataset"], init_cls, args['increment'], args['file_name'])

    log_suffix = '{}_{}_{}_{}_4lamda:{}_{}_{}_{}_{}_task:{}_backbond:{}_prefix:{}_task:{}'.format(args["prefix"], args["seed"], args["convnet_type"],
                                                                              args['vpt_type'], args['lamda_for_logit_loss'], args['lamda_for_prompt'],
                                                                                       args['lamda_for_featureformer'], args['lamda_for_featurelower'], args['lamda_for_pool3'],
                                                                                          args['task_num'], backbone, 1, args['task_id'])

    prompt_suffix = '{}'.format(args['task_id'])

    logfilename = logs_name + log_suffix

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s [%(filename)s] => %(message)s",
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if args['use_tensorboard'] == 1:
        tensorboard_logdir = logs_name + 'tensorboard_logs/'
        if not  os.path.exists(tensorboard_logdir):
            os.makedirs(tensorboard_logdir)
        WRITER = SummaryWriter(log_dir=tensorboard_logdir + log_suffix)
    else:
        WRITER = None

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args['task_id']
    )

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    running_prompt_token_list = []
    current_task_num = 0
    for task in range(data_manager.nb_tasks):

        current_task_num += 1

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager, WRITER)


        # visualization of prompt token
        # current_prompt = model._network.convnets[1].Prompt_Tokens
        # running_prompt_token_list.append(current_prompt)
        #
        # with_init = (current_prompt ** 2).sum() / (current_prompt.shape[0] * current_prompt.shape[1] * current_prompt.shape[2])
        # WRITER.add_scalar('Prompt_Token/with_init', with_init, global_step=current_task_num)

        # if current_task_num > 1:
        #     external_list = []
        #     for i in range(len(running_prompt_token_list) - 1):
        #         external_list.append(
        #             ( (running_prompt_token_list[i] - current_prompt) ** 2).sum()
        #         )
        #     index = external_list.index(max(external_list)) + 1
        #     WRITER.add_scalar('Prompt_Token/max_familiar_index', index, global_step=current_task_num)

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            cnn_out = "CNN: {}".format(cnn_accy["grouped"])
            logger_list.append(cnn_out)
            logger_list_new.append(cnn_accy['grouped']['new'])
            logger_list_old.append(cnn_accy['grouped']['old'])

            logging.info(cnn_out)
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            cnn_top1_out = "CNN top1 curve: {}".format(cnn_curve["top1"])
            logging.info(cnn_top1_out)
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            acc_out = "Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info(acc_out)
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            cnn_out = "CNN: {}".format(cnn_accy["grouped"])
            logger_list.append(cnn_out)
            logging.info(cnn_out)
            logger_list_new.append(cnn_accy['grouped']['new'])
            logger_list_old.append(cnn_accy['grouped']['old'])

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            cnn_top1_out = "CNN top1 curve: {}".format(cnn_curve["top1"])
            logging.info(cnn_top1_out)
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            acc_out = "Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info(acc_out)

    if args['prompt_store'] == '':
        # no storing
        pass
    else:
        prompt_param = model._network.convnets[1].Prompt_Tokens
        torch.save(prompt_param, logs_name + prompt_suffix + 'prompt_weight.pt')

    logging.info('\n')
    logging.info('===========================================')
    logging.info(' ================>  Setting  <================ ')

    logging.info('lamda_for_featureformer : {}'.format(args['lamda_for_featureformer']))
    logging.info('lamda_for_pool3 : {}'.format(args['lamda_for_pool3']))
    logging.info('lamda_for_prompt : {}'.format(args['lamda_for_prompt']))
    logging.info('fc_init_type : {}'.format(args['fc_inittype']))

    logging.info('\n')

    logging.info('lamda_for_featurelower : {}'.format(args['lamda_for_featurelower']))
    logging.info('lamda_for_logit_loss : {}'.format(args['lamda_for_logit_loss']))

    logging.info('feature_distill_type : {}'.format(args['feature_distill_type']))
    logging.info('task_num : {}'.format(args['task_num']))
    logging.info('intra_share : {}'.format(args['intra_share']))
    logging.info('prompt_token_num : {}'.format(args['prompt_token_num']))
    logging.info("used_decouple : {} and decouple_type : {}".format(args["used_decouple"], args["decouple_type"]))

    if args['decouple_type'] == "Block":
        logging.info("Block_which : {}".format(args["Block_which"]))
    if args['decouple_type'] == "Part":
        logging.info("Part_which : {}".format(args["Part_which"]))

    logging.info('===========================================')
    logging.info('\n')
    for i in logger_list:
        logging.info(i)

    logging.info(cnn_top1_out)
    logging.info(acc_out)

    logging.info('new : {}'.format(logger_list_new))
    logging.info('old : {}'.format(logger_list_old))

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
