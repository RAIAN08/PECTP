import json
import argparse
import random
from trainer import train
from utils.parser_utils import * 

def main():
    args = setup_parser().parse_args()

    # config
    print(args.config)
    args.config= './exps/{}/adam_vpt_deep.json'.format(args.dataset_name)
    print(args.config)
    param = load_json(args.config)

    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    args = fix_params_from_parser(args)
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


#===============================================================
def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='', help='Json file of settings.')
    parser.add_argument('--dataset_name', type=str, default='', help='Json file of settings.')

    parser.add_argument('--task_num', type=int, default=10, help='how many change times when tuning?') # 10
    parser.add_argument('--gpu_used', type=int, default=999, help='chose gpu id if not 999')
    parser.add_argument('--prompt_num', type=int, default=999, help='chose prompt num if not 999')
    parser.add_argument('--use_tensorboard', type=int, default=0, help='chose use tensorboard or not?')

    parser.add_argument('--feature_distill_type', type=str, default='allheadallfeature', help='chose feature distill type?'
                        '---allheadallfeature   allheadpromptfeature    lastheadpromptfeature || default is 0。') # ''

    parser.add_argument('--file_name', type=str, default='', help='change the name of log file, only to see distinctly...')

    parser.add_argument('--lamda_for_logit_loss', type=str, default=0, help='chose use tensorboard or not?') # 0
    parser.add_argument('--lamda_for_prompt', type=str, default=0, help='chose use tensorboard or not?') # 0
    parser.add_argument('--lamda_for_featureformer', type=str, default=0, help='chose use tensorboard or not?') # 0
    parser.add_argument('--lamda_for_featurelower', type=str, default=0, help='chose use tensorboard or not?') # 0
    parser.add_argument('--lamda_for_pool3', type=str, default=0, help='chose use tensorboard or not?') # 0

    parser.add_argument('--intra_share', type=int, default=1, help='chose to intra share or not? default is share!') # 1

    parser.add_argument('--used_decouple', type=int, default=0, help='decouple the loss or not? default is no!') # 1
    parser.add_argument('--decouple_type', type=str, default="", help=' decouple type ? Block or Part? default is Nothing!')
    parser.add_argument('--Block_which', type=str, default="", help='which block or blocks fetch from......when use decouple type with Block...!') # 1
    parser.add_argument('--Part_which', type=str, default="", help='which part or parts fetch from......when use decouple type with Part...!') # 1

    parser.add_argument('--ratio_fix_para', type=int, default=0, help='to auto sets the scale of feature loss with the ratio para, after fetch the exper ratio between cls and feature loss.......')


    parser.add_argument('--fc_inittype', type=str, default="type8", help="chose init type of fc......... corresponding to yueque text, type8, type7, type6, type5") #

    parser.add_argument('--train_list', type=str, default='', help='...')

    parser.add_argument('--prompt_store', type=str, default='', help='store the prompt paramter?')
    parser.add_argument('--task_id', type=int, default=0,  help='save the num of the task...?')

    parser.add_argument('--fusion_type', type=str, default='concat', help='pointadd, concat, no_fusion, continual_extract')

    return parser
#===============================================================

if __name__ == '__main__':
    main()
