import argparse
import random

def fix_params_from_parser(args):
    # gpu used
    if args['gpu_used'] == 999:
        pass
    else:
        GPU_ID = ["{}".format(args['gpu_used'])]
        args["device"] = GPU_ID

    # prompt_num
    if args['prompt_num'] == 999:
        pass
    else:
        PROMPT_NUM = args['prompt_num']
        args['prompt_token_num'] = PROMPT_NUM

    if args['Block_which'] == "":
        pass
    else:
        real_block_which = list(map(int, args['Block_which'].split()))
        args['Block_which'] = real_block_which

    if args['Part_which'] == "":
        pass
    else:
        real_part_which = list(map(int, args['Part_which'].split()))
        args['Part_which'] = real_part_which

    if args['lamda_for_logit_loss'] == '':
        pass
    else:
        args['lamda_for_logit_loss'] = float(args['lamda_for_logit_loss'])


    if args['lamda_for_prompt'] == '':
        pass
    else:
        args['lamda_for_prompt'] = float(args['lamda_for_prompt'])

    if args['lamda_for_featureformer'] == '':
        pass
    else:
        args['lamda_for_featureformer'] = float(args['lamda_for_featureformer'])

    if args['lamda_for_featurelower'] == '':
        pass
    else:
        args['lamda_for_featurelower'] = float(args['lamda_for_featurelower'])

    if args['lamda_for_pool3'] == '':
        pass
    else:
        args['lamda_for_pool3'] = float(args['lamda_for_pool3'])
    
    args['file_name'] = 'Temporary_Log' + '{}'.format(random.randint(10000, 20000)) if args['file_name'] == '' else args['file_name']

    args['print'] = 1

    return args