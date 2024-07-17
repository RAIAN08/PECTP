def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name=="adam_finetune":
        from models.adam_finetune import Learner
        return Learner(args)
    elif name=="adam_ssf":
        from models.adam_ssf import Learner
        return Learner(args)
    elif name=="adam_vpt":
        if args['fusion_type'] != 'no_fusion':
            from models.adam_vpt import Learner
        else:
            from models.adam_vpt_nofusion import  Learner
        return Learner(args) 
    elif name=="adam_adapter":
        from models.adam_adapter import Learner
        return Learner(args)
    else:
        assert 0
