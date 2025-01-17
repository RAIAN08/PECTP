import math
import timm
import torch
import torch.nn as nn
# from functools import partial
# from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
# from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Attention, Block
# from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
#     resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked

from vit_model.vit_tfm import VisionTransformer, PatchEmbed


def build_promptmodel(modelname='vit_base_patch16_224',  Prompt_Token_num=10, VPT_type="Deep", args=None):
    # VPT_type = "Deep" / "Shallow"
    edge_size=224
    patch_size=16  
    num_classes=1000 if modelname == 'vit_base_patch16_224' else 21843

    # fc = CosineLinear(10, 768)

    basic_model = timm.create_model(modelname, pretrained=True)
    model = VPT_ViT(Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type, args=args)
    # model.New_CLS_head(num_classes)

    # fc = CosineLinear(10, 768)

    # drop head.weight and head.bias
    basicmodeldict=basic_model.state_dict()
    basicmodeldict.pop('head.weight')
    basicmodeldict.pop('head.bias')

    model.load_state_dict(basicmodeldict, False)
    model.head = torch.nn.Identity()
    model.Freeze()
    
    return model


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None, args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size,
                         in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer, Prompt_Token_num=Prompt_Token_num)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.Prompt_Token_num = Prompt_Token_num
        self.VPT_type = VPT_type
        self.args = args

        # todo:rewrite.......which only for feature store
        self.tfmout = []

        # use different types to init prompts
        if self.args['intra_share'] == 1:
            # da-wei, a type of intra-shared layer
            print('use intra share to init prompt!')
            if VPT_type == "Deep":
                self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
            else:  # "Shallow"
                self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
        else:
            # vpt type: all with random init from U-dist (implemtation from vpt)
            # first get val
            # todo: to be comfirmed the implemtation of un intra share...
            val = math.sqrt(6. / float(768 * 2))
            if VPT_type == "Deep":
                self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
            else:  # "Shallow"
                self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
            nn.init.uniform_(self.Prompt_Tokens, -val, val)
            print('use un intra share to init prompt!')
        # fc = CosineLinear(10, 768)

        # fc = CosineLinear(10, 768)

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def forward_features(self, x):
        if len(self.tfmout) != 0:
            self.tfmout = []
        self.B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x, attn_softmax = self.blocks[i](x)

                self.tfmout.append(x)

                x = x[:,:num_tokens - Prompt_Token_num]
        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x, attn_softmax

    def forward(self, x):
        self.tfmout = [] # renew

        x, attn_softmax = self.forward_features(x)

        # use cls token for cls head
        # x = self.pre_logits(x[:, 0, :])
        x=x[:, 0, :]

        # if attn_softmax != None:
        #     attn_softmax = attn_softmax.view(self.B, -1, attn_softmax.shape[2], attn_softmax.shape[2])


        # x = self.head(x)
        return x, attn_softmax

    def get_each_tfmout(self, x):
        tfmout = self.tfmout
        self.tfmout = [] # renew
        return tfmout