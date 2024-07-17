import logging
import numpy as np
import torch
import copy
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet, SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from functools import reduce

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        print(">>>>>>>    nof fusion module.........")
        if 'vpt' not in args["convnet_type"]:
            raise NotImplementedError('VPT requires VPT backbone')

        if 'resnet' in args['convnet_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size = 128
            self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        else:
            self._network = SimpleVitNet(args, True)
            self.batch_size = args["batch_size"]
            self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8

        self.print1 = 1
        self.print2 = 1

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc2_for_infer(self, trainloader, model, args):

        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self.args["device"][0])
                label = label.to(self.args["device"][0])
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            model.fc2.weight.data[class_index] = proto
            # self._network.fc.weight.data[class_index] = proto
        return model

    def replace_fc(self, trainloader, model, args):

        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self.args["device"][0])
                label = label.to(self.args["device"][0])
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc2.weight.data[class_index] = proto
        return model



    def _get_distill_loss(self, old_logits, new_logits, soft_T):
        soft_logits = torch.softmax(old_logits / soft_T, dim=1)
        new_logits = torch.log_softmax(new_logits / soft_T, dim=1)
        return -1 * torch.mul(soft_logits, new_logits).sum() / old_logits.shape[0]

    def _get_featuremap_from_TandS(self):
        old_feature = self._old_network.convnet.get_each_tfmout()
        new_feature = self._network.convnet.get_each_tfmout()

        old_feature = torch.stack(old_feature)
        new_feature = torch.stack(new_feature)

        return old_feature, new_feature

    def _get_blockandpart(self, teacher, student):
        # L2
        uuu = (teacher - student) ** 2

        # loss_each_block = torch.sum(uuu, dim=(1, 2, 3)) / (reduce(lambda x, y: x * y, teacher.shape))
        loss_each_block = torch.sum(uuu, dim=(1, 2, 3))

        ppp = torch.sum(uuu, dim=1)
        uuu_cls, uuu_general, uuu_prompt= ppp[:, 0:1, :], ppp[:, 1:197, :], ppp[:, 197:, :]
        loss_each_part = [uuu_cls.sum() / len(uuu_cls), uuu_general.sum() / len(uuu_general), uuu_prompt.sum() / len(uuu_prompt)]

        return loss_each_block, loss_each_part

    def _get_feature_from_each_block_loss(self, soft_T):
        # print(len(self._old_vptfc.convnet.tfmout), len(self._network.convnet.tfmout))

        old_feature = self._old_vptfc.convnet.get_each_tfmout()
        new_feature = self._network.convnet.get_each_tfmout()

        old_feature = torch.stack(old_feature)
        new_feature = torch.stack(new_feature)


        uuu = (new_feature - old_feature)
        if self.args['feature_distill_type'] == 'allheadallfeature':
            uuu = uuu ** 2
            loss_each_block = torch.sum(uuu, dim=(1, 2, 3)) / (old_feature.shape[0] * old_feature.shape[1])

            ppp = torch.sum(uuu, dim=1)
            uuu_cls = ppp[:, 0:1, :].sum() / (old_feature.shape[0] * old_feature.shape[1])
            uuu_general = ppp[:, 1:197, :].sum() / (old_feature.shape[0] * old_feature.shape[1])
            uuu_prompt = ppp[:, 197:, :].sum() / (old_feature.shape[0] * old_feature.shape[1])
            loss_each_part = [uuu_cls, uuu_general, uuu_prompt]

            if self.args['used_decouple']:
                adj_loss = self.decouple_loss(uuu, loss_each_block, loss_each_part)
                uuu = adj_loss
            else:
                # uuu = uuu.sum() / (old_feature.shape[0] * old_feature.shape[1])
                # uuu = uuu.sum() / (old_feature.shape[0] * old_feature.shape[1] * old_feature.shape[2] * old_feature.shape[3])
                uuu = uuu.sum() / (old_feature.shape[0] * old_feature.shape[1] * old_feature.shape[3])

            if self.print1 == 1:
                logging.info('used allheadallfeature loss ')
                self.print1 = 0

        elif self.args['feature_distill_type'] == 'allheadpromptfeature':
            uuu = uuu[:, :, -self.args['prompt_token_num']:, :]
            uuu = uuu ** 2
            uuu = uuu.sum() / (old_feature.shape[0] * old_feature.shape[1])
            if self.print1 == 1:
                logging.info('used allheadpromptfeature loss ')
                self.print1 = 0
            loss_each_block = loss_each_part = []
            # todo:...mopdify the scale of uuu

        elif self.args['feature_distill_type'] == 'lastheadpromptfeature':
            uuu = uuu[-1:, :, -self.args['prompt_token_num']:, :]
            uuu = uuu ** 2
            uuu = uuu.sum() / (old_feature.shape[0] * old_feature.shape[1])
            if self.print1 == 1:
                logging.info('used lastheadpromptfeature loss ')
                self.print1 = 0
            loss_each_block = loss_each_part = []
            # todo:...mopdify the scale of uuu
        else:
            uuu = 0
            loss_each_block = loss_each_part = []
        return uuu, (loss_each_block, loss_each_part)

    def decouple_loss(self, input, each_block, each_part):

        all_block_loss = each_block
        all_part_loss = each_part
        Decouple_Type = self.args['decouple_type']

        if Decouple_Type == "Block":
            which_block = self.args['Block_which']
            select_block_loss = all_block_loss[which_block]
            return select_block_loss.sum()
        elif Decouple_Type == "Part":
            which_part = self.args['Part_which']
            select_part_loss = [all_part_loss[i] for i in which_part]
            return sum(select_part_loss)
        else:
            print("NO TYPE  ==========================================")
            return 0

    def assemble_loss(self, batch_loss_cls, batch_loss_logit, batch_loss_feature, batch_prompt_loss):

        if self.Auto_Ratio == "Auto":
            real_bs_loss_logit = 0
            real_bs_prompt_loss = 0
            if batch_loss_feature.item() == 0:
                real_bs_loss_feature = batch_loss_feature
                # logging.info("========================================================================== > meet 0=====")
            else:
                auto_lamda_feature = batch_loss_cls.item() / (self.ratio_fix_para * batch_loss_feature.item())
                real_bs_loss_feature = batch_loss_feature * auto_lamda_feature

        else:
            lamda1 = self.args['lamda_for_logit_loss'] if self.args['lamda_for_logit_loss'] != 0 else 0
            lamda2 = self.args['lamda_for_feature_loss'] if self.args['lamda_for_feature_loss'] != 0 else 0
            lamda3 = self.args['lamda_for_prompt'] if self.args['lamda_for_prompt'] != 0 else 0

            real_bs_loss_logit = lamda1 * batch_loss_logit if lamda1 != 0 else 0
            real_bs_loss_feature = lamda2 * batch_loss_feature if lamda2 != 0 else 0
            real_bs_prompt_loss = lamda3 * batch_prompt_loss if lamda3 != 0 else 0

        return batch_loss_cls, real_bs_loss_logit, real_bs_loss_feature, real_bs_prompt_loss

    def incremental_train(self, data_manager, writer):
        self.writer = writer
        self.Auto_Ratio = 'Auto' if self.args['ratio_fix_para'] != 0 else 'Handmaking'
        self._cur_task += 1
        self.incre_task = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.incre_task

        self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)

        if self._cur_task < self.args['task_num']:
            KD_switch = False if self._cur_task == 0 else True  # wherher to kd depend on whcih stage is among the incremental learning....

            # Freeze the parameters for ViT.
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

            # if some parameters are trainable, print the key name and corresponding parameter number
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            # optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)

            if self.args['train_list'] != '':
                if self._cur_task in self.args['train_list']:

                    self._init_train(train_loader, test_loader, optimizer, scheduler, KD_switch)
                else:
                    pass
            else:
                self._init_train(train_loader, test_loader, optimizer, scheduler, KD_switch)

                self._old_network = copy.deepcopy(self._network)

                self._network.update_fc2_for_infer(self._known_classes + self.incre_task)

        else:
            pass

        self.replace_fc(train_loader_for_protonet, self._network, None)

    # def construct_dual_branch_network(self):
    #     network = MultiBranchCosineIncrementalNet(self.args, True)
    #     network.construct_dual_branch_network(self._network)
    #     self._network = network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, KD_switch):
        if KD_switch:
            # means need to calculate kd loss:
            logging.info('=============== >>> train with both kd loss and cls loss <<< ====================')
            self._init_train_with_kd(train_loader, test_loader, optimizer, scheduler)
        else:
            # means no need to calculate kd loss:
            logging.info('=============== >>> train without kd loss, but only with cls loss <<< ====================')
            self._init_train_without_kd(train_loader, test_loader, optimizer, scheduler)

    def _init_train_without_kd(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_cls = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                fake_targets = targets - self._known_classes
                if self.args['fc_inittype'] == 'type7' or self.args['fc_inittype'] == 'type5':
                    batch_loss_cls = F.cross_entropy(logits, targets)
                elif self.args['fc_inittype'] == 'type8' or self.args['fc_inittype'] == 'type6':
                    batch_loss_cls = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                else:
                    pass

                loss = batch_loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_cls += batch_loss_cls.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if self.args['use_tensorboard'] == 1:
                writer = self.writer
                writer.add_scalar('Loss/loss', losses / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                writer.add_scalar('Loss/loss_cls', losses_cls / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _init_train_with_kd(self, train_loader, test_loader, optimizer, scheduler):

        if self.Auto_Ratio == 'Auto':
            self.ratio_fix_para = self.args['ratio_fix_para']
            logging.info('Auto making lamda with fixed ratio...')
        else:
            logging.info('Hand making lamda...')

        lamda1 = self.args['lamda_for_logit_loss'] if self.args['lamda_for_logit_loss'] != 0 else 0
        lamda2 = self.args['lamda_for_prompt'] if self.args['lamda_for_prompt'] != 0 else 0
        lamda3 = self.args['lamda_for_featureformer'] if self.args['lamda_for_featureformer'] != 0 else 0
        lamda4 = self.args['lamda_for_featurelower'] if self.args['lamda_for_featurelower'] != 0 else 0
        lamda5 = self.args['lamda_for_pool3'] if self.args['lamda_for_pool3'] != 0 else 0

        print("##========== LAMDA ================\n")
        print("lamda for logits : {}".format(lamda1))
        print("lamda for prompt : {}".format(lamda2))
        print("lamda for featureformer : {}".format(lamda3))
        print("lamda for featurelower : {}".format(lamda4))
        print("lamda_for_pool3 : {}\n".format(lamda5))

        print('train_list : {}'.format(self.args['train_list']))
        print("\n")
        print("##========== LAMDA ================")

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses, losses_cls, losses_logit, losses_feature_former, losses_feature_lower, losses_feature_pool3, losses_prompt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            losses_each_block = np.zeros(12)
            losses_each_part = np.zeros(3)

            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                fake_targets = targets - self._known_classes
                if self.args['fc_inittype'] == 'type7' or self.args['fc_inittype'] == 'type5':
                    batch_loss_cls = F.cross_entropy(logits, targets)
                elif self.args['fc_inittype'] == 'type8' or self.args['fc_inittype'] == 'type6':
                    batch_loss_cls = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                else:
                    pass

                soft_T = 2
                with torch.no_grad():
                    old_pre_logits = self._old_network(inputs)
                    old_logits = old_pre_logits['logits']
                batch_loss_logit = self._get_distill_loss(old_logits, logits[:, :self._known_classes], soft_T) * lamda1
                del old_logits

                teacher_featuremap, student_featuremap = self._get_featuremap_from_TandS()

                batch_loss_each_block, batch_loss_each_part = self._get_blockandpart(teacher_featuremap, student_featuremap)

                batch_loss_feature_form, batch_loss_feature_lower, batch_loss_pool3 = self._network.ViTKDLoss.forward(
                    [student_featuremap[:6, :, :, :], student_featuremap[6:, :, :, :]],
                    [teacher_featuremap[:6, :, :, :], teacher_featuremap[6:, :, :, :]],
                    lamda3, lamda4, lamda5
                )

                del student_featuremap, teacher_featuremap

                prompt_shape = self._network.convnet.Prompt_Tokens.shape
                batch_prompt_loss = ((self._network.convnet.Prompt_Tokens - self._old_network.convnet.Prompt_Tokens) ** 2).sum() / (prompt_shape[0] * prompt_shape[1] * prompt_shape[2]) * lamda2

                loss = batch_loss_cls + \
                       batch_loss_logit + \
                       batch_loss_feature_form + batch_loss_feature_lower + batch_loss_pool3 + \
                       batch_prompt_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_cls += batch_loss_cls.item()
                if isinstance(batch_loss_logit, torch.Tensor):
                    losses_logit += batch_loss_logit.item()

                if isinstance(batch_loss_feature_form, torch.Tensor):
                    losses_feature_former += batch_loss_feature_form.item()
                if isinstance(batch_loss_feature_lower, torch.Tensor):
                    losses_feature_lower += batch_loss_feature_lower.item()
                if isinstance(batch_loss_pool3, torch.Tensor):
                    losses_feature_pool3 += batch_loss_pool3.item()

                if isinstance(batch_prompt_loss, torch.Tensor):
                    losses_prompt += batch_prompt_loss.item()

                if len(batch_loss_each_block) != 0:
                    losses_each_block += batch_loss_each_block.cpu().detach().numpy()
                    del batch_loss_each_block
                if len(batch_loss_each_part) != 0:
                    loss_each_part_numpy = []
                    for item in batch_loss_each_part:
                        loss_each_part_numpy.append(item.cpu().detach().numpy())
                    loss_each_part_numpy = np.array(loss_each_part_numpy)
                    losses_each_part += loss_each_part_numpy
                    del batch_loss_each_part

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # tensorboard..sets
            if self.args['use_tensorboard'] == 1:
                writer = self.writer
                writer.add_scalar('Loss/loss', losses / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                writer.add_scalar('Loss/loss_cls', losses_cls / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                writer.add_scalar('Loss/loss_logit', losses_logit / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)

                writer.add_scalar('Loss/loss_feature_former', losses_feature_former / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                writer.add_scalar('Loss/loss_feature_lower', losses_feature_lower / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                writer.add_scalar('Loss/loss_feature_pool3', losses_feature_pool3 / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)

                writer.add_scalar('Loss/loss_prompt', losses_prompt / len(train_loader),
                                  global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)

                if self._cur_task != 0:
                    for i in range(len(losses_each_block)):
                        writer.add_scalar('loss_each_block/block : {}'.format(i),
                                          losses_each_block[i] / len(train_loader),
                                          global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)

                if self._cur_task != 0:
                    for j in range(len(losses_each_part)):
                        writer.add_scalar('loss_each_part/part : {}'.format(j), losses_each_part[j] / len(train_loader),
                                          global_step=(self._cur_task) * self.args['tuned_epoch'] + epoch)
                        
            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

