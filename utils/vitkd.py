import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTKDLoss(nn.Module):
    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 alpha_vitkd=0.00003,
                 beta_vitkd=0.000003,
                 lambda_vitkd=0.5,
                 ):
        super(ViTKDLoss, self).__init__()
        self.alpha_vitkd = alpha_vitkd
        self.beta_vitkd = beta_vitkd
        self.lambda_vitkd = lambda_vitkd

        # if student_dims != teacher_dims:
        #     self.align2 = nn.ModuleList([
        #         nn.Linear(student_dims, teacher_dims, bias=True)
        #         for i in range(2)])
        #     self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        # else:
        #     self.align2 = None
        #     self.align = None

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))

        # self.generation = nn.Sequential(
        #     nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))
    # def forward(self,
    #             preds_S,
    #             preds_T):
    #     """Forward function.
    #     Args:
    #         preds_S(List): [B*2*N*D, B*N*D], student's feature map
    #         preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
    #     """
    #     low_s = preds_S[0]
    #     low_t = preds_T[0]
    #     high_s = preds_S[1]
    #     high_t = preds_T[1]
    #
    #     B = low_s.shape[0]
    #     loss_mse = nn.MSELoss(reduction='sum')
    #
    #     '''ViTKD: Mimicking'''
    #     # if self.align2 is not None:
    #     #     for i in range(2):
    #     #         if i == 0:
    #     #             xc = self.align2[i](low_s[:, i]).unsqueeze(1)
    #     #         else:
    #     #             xc = torch.cat((xc, self.align2[i](low_s[:, i]).unsqueeze(1)), dim=1)
    #     # else:
    #     xc = low_s
    #
    #     loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd
    #
    #     '''ViTKD: Generation'''
    #     # if self.align is not None:
    #     #     x = self.align(high_s)
    #     # else:
    #     x = high_s
    #
    #     # Mask tokens
    #     B, N, D = x.shape
    #     x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
    #     mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
    #     x_ = torch.cat([x, mask_tokens], dim=1)
    #     x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
    #     mask = mat.unsqueeze(-1)
    #
    #     hw = int(N ** 0.5)
    #     x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
    #     x = self.generation(x).flatten(2).transpose(1, 2)
    #
    #     loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
    #     loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd
    #
    #     return loss_lr + loss_gen

    def forward1(self, preds_S, preds_T, lamda3, lamda4, lamda5):
        """Forward function.
        Args:
            preds_S(List): [low_head*B*N*D, high_head*B*N*D], student's feature map
            preds_T(List): [low_head*B*N*D, low_head*B*N*D], teacher's feature map
        """

        '''ViTKD: Mimicking'''

        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]

        B = low_s.shape[1]
        loss_mse = nn.MSELoss(reduction='sum')

        xc = low_s

        # loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd
        loss_lr = loss_mse(xc, low_t) / B * (1/(12 * lamda3)) if lamda3 != 0 else 0

        '''ViTKD: Generation'''
        x = high_s
        loss_gen = loss_mse(x, high_t) / B * (1/(12 * lamda4)) if lamda4 != 0 else 0

        return loss_lr, loss_gen, 0

    def forward(self, preds_S, preds_T, lamda3, lamda4, lamda5):
        low_s = preds_S[0]
        low_t = preds_T[0]
        high_s = preds_S[1]
        high_t = preds_T[1]

        B = low_s.shape[1] # batch
        loss_mse = nn.MSELoss(reduction='sum') # loss func

        # concate first, shape: [all_head, B, N, D]
        teacher_orign = torch.concat((low_t, high_t), dim=0).permute(1, 0, 2, 3) # shape: [b, head, n, d]
        student_orign = torch.concat((low_s, high_s), dim=0).permute(1, 0, 2, 3)

        # pool on 3 dim,
        # pool on head
        teacher_pool_head = torch.sum(teacher_orign, dim=1)
        student_pool_head = torch.sum(student_orign, dim=1)


        # pool on patch
        teacher_pool_patch = torch.sum(teacher_orign, dim=2) # shape : [bs, head=block_num, dim]
        student_pool_patch = torch.sum(student_orign, dim=2)


        # pool on dim
        teacher_pool_dim = torch.sum(teacher_orign, dim=3) # shape : [bs, head=block_num, patch_num]
        student_pool_dim = torch.sum(student_orign, dim=3)

        loss_1 = loss_mse(teacher_pool_head, student_pool_head) / B * (1/(12 * lamda3)) if lamda3 != 0 else 0
        loss_2 = loss_mse(teacher_pool_patch, student_pool_patch) / B * (1/(12 * lamda4)) if lamda4 != 0 else 0 # out of usage
        loss_3 = loss_mse(teacher_pool_dim, student_pool_dim) / B * (1/(12 * lamda5)) if lamda5 != 0 else 0

        return loss_1, loss_2, loss_3


    def forward2(self, preds_S, preds_T, lamda3, lamda4):
        pass


    # low_s = preds_S[0]
        # low_t = preds_T[0]
        # high_s = preds_S[1]
        # high_t = preds_T[1]
        #
        # B = low_s.shape[0]
        # loss_mse = nn.MSELoss(reduction='sum')
        #
        # '''ViTKD: Mimicking'''
        # # if self.align2 is not None:
        # #     for i in range(2):
        # #         if i == 0:
        # #             xc = self.align2[i](low_s[:, i]).unsqueeze(1)
        # #         else:
        # #             xc = torch.cat((xc, self.align2[i](low_s[:, i]).unsqueeze(1)), dim=1)
        # # else:
        # xc = low_s
        #
        # loss_lr = loss_mse(xc, low_t) / B * self.alpha_vitkd
        #
        # '''ViTKD: Generation'''
        # # if self.align is not None:
        # #     x = self.align(high_s)
        # # else:
        # x = high_s
        #
        # # Mask tokens
        # B, N, D = x.shape
        # x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
        # mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        # x_ = torch.cat([x, mask_tokens], dim=1)
        # x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        # mask = mat.unsqueeze(-1)
        #
        # hw = int(N ** 0.5)
        # x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        # x = self.generation(x).flatten(2).transpose(1, 2)
        #
        # loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        # loss_gen = loss_gen / B * self.beta_vitkd / self.lambda_vitkd

        # return loss_lr + loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked