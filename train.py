import torch
# 更新 import 语句
from torch.cuda.amp import GradScaler, autocast
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import set_seed, AverageMeter
import os
# from torch import bmm
from transformers import CLIPProcessor, CLIPModel, Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2VLImageProcessor
from torch.optim.lr_scheduler import LinearLR
from qwen_vl_utils import process_vision_info
from open_clip import create_model_from_pretrained, get_tokenizer

scaler = GradScaler()

# define a function to log the training process
def log_train(epoch, loss, acc):
    print(f"epoch {epoch}, loss {loss}, acc {acc}")
    with open('train.log', 'a') as f:
        f.write(f"epoch {epoch}, loss {loss}, acc {acc}\n")

# define a function to log the testing process
def log_test(epoch, acc):
    print(f"epoch {epoch}, acc {acc}")
    with open('test.log', 'a') as f:
        f.write(f"epoch {epoch}, acc {acc}\n")

def get_clip_vision_model(model_name):
    model = CLIPModel.from_pretrained(model_name).vision_model
    return model

def get_openclip_vit_big_G():
    # this is the vision encoder used by Qwen-VL
    model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    # write config to a file
    # with open('openclip_vit_bigG_config.log', 'a') as f:
    #     f.write(str(model.config))
    return model.visual

def get_dfn_vision_model():
    # this is the vision encoder used by Qwen2-VL, but with notable modifications
    print(f"getting dfn vision model")
    model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
    return model.visual

def get_qwen2vl_vision_encoder():
    print(f"getting qwen2vl vision encoder")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    # with open('qwenconfig', 'a') as f:
    #     f.write(str(model.visual.config))
    import copy
    backbone = copy.deepcopy(model.visual)
    
    # 
    del model
    torch.cuda.empty_cache()  # release memory
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return backbone

def get_qwenvl_vision_encoder():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
    # print(model)
    # write the model structure to a file
    # with open('qwen_vl_model.log', 'a') as f:
    #      f.write(str(model))
    # write config to a file
    with open('qwen_vl_config.log', 'a') as f:
        f.write(str(model.config))
    
    # inspect the forward function
    # import inspect
    # with open('qwen_vl_forward.log', 'a') as f:
    #     f.write(inspect.getsource(model.transformer.visual.__init__))
    return model.transformer.visual


def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            # print(x.shape, y.shape)
            opt.zero_grad()
            with autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ep % 10 == 9:
            acc = test(vit, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
            pbar.set_description('best_acc ' + str(args.best_acc))

    model = model.cpu()
    return model

def train_qwen2(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    # 更新 GradScaler 的初始化
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y, thw = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            thw = thw.squeeze(1)
            # print(x.shape, y.shape, thw.shape)
            opt.zero_grad()
            with autocast():
                out = model(x, thw=thw)
                # print(out.shape)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ep % 10 == 9:
            acc = test(vit, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
            pbar.set_description('best_acc ' + str(args.best_acc))

    model = model.cpu()
    return model

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        # 更新 autocast 的用法
        with autocast():
            out = model(x).data
        acc.update(out, y)
    return acc.result().item()

class ViTLargeInitFromOpenAICliP(nn.Module):
    def __init__(self, backbone, class_number):
        super().__init__()
        self.backbone = backbone
        self.class_number = class_number
        self.classifier = nn.Linear(1024, class_number)
    def forward(self, x):
        # x = self.backbone(x).pooler_output
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ViTBigGInitFromOpenCliP(nn.Module):
    def __init__(self, backbone, class_number):
        super().__init__()
        self.backbone = backbone
        self.class_number = class_number
        self.classifier = nn.Linear(1280, class_number)
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ViTBigGInitFromQwenVL(nn.Module):
    def __init__(self, backbone, class_number):
        super().__init__()
        self.backbone = backbone
        self.class_number = class_number
        self.classifier = nn.Linear(4096, class_number)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ViTInitFromQwenVL(nn.Module):
    def __init__(self, backbone, class_number):
        super().__init__()
        self.backbone = backbone
        self.class_number = class_number
        # self.classifier = nn.Linear(1024, class_number)
        self.classifier = nn.Linear(1280, class_number)
    def forward(self, x, thw):
        # x = self.backbone(x).pooler_output
        x = self.backbone(x, grid_thw=thw)
        x = self.classifier(x)
        return x

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def print_params(model):
    for name, param in model.named_parameters():
        print(name)
        with open('params.log', 'a') as f:
            f.write(name + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--b_lr', type=float, default=5e-5)
    parser.add_argument('--h_lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--model', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--dataset', type=str, default='caltech101')
    parser.add_argument('--mm_trained', action='store_true', default=False)
    parser.add_argument('--qwen2', action='store_true', default=False)
    parser.add_argument('--qwen', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    args.best_acc = 0

    # define the model and dataloader
    if not args.mm_trained:
        # backbone = get_dfn_vision_model()
        backbone = get_openclip_vit_big_G()
    else:
        # backbone = get_qwen2vl_vision_encoder()
        backbone = get_qwenvl_vision_encoder()
    if not args.qwen2 and not args.qwen:
        print(f"using original clip models. ")
        # vit = ViTLargeInitFromOpenAICliP(backbone, get_classes_num(args.dataset))
        vit = ViTBigGInitFromOpenCliP(backbone, get_classes_num(args.dataset))
        train_dl, test_dl = get_data(args.dataset, normalize=True)
    elif args.qwen:
        print(f"using vision encoder from qwen vl.")
        vit = ViTBigGInitFromQwenVL(backbone, get_classes_num(args.dataset))
        train_dl, test_dl = get_data_for_qwen(args.dataset)
    else:
        print(f"using vision encoder from qwen2 vl.")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        vit = ViTInitFromQwenVL(backbone, get_classes_num(args.dataset))
        train_dl, test_dl = get_data_for_qwen2(args.dataset, normalize=False)
    
    # train_dl, test_dl = get_data(args.dataset, normalize=False)
    print(sum(p.numel() for p in vit.parameters()))
    # print_params(vit)
    vit.backbone.requires_grad_(False)
    vit.classifier.requires_grad_(True)
    print(f"param is {vit.count_trainable_params()}")
    print(f"training on dataset {args.dataset}")

    # define optimizer and scheduler
    opt = AdamW([
    # {'params': vit.backbone.parameters(), 'lr': args.b_lr, 'weight_decay': args.wd},
    {'params': vit.classifier.parameters(), 'lr': args.h_lr, 'weight_decay': args.wd}
    ])
    # scheduler = CosineLRScheduler(opt, t_initial=100,
    #                                 warmup_t=10, lr_min=1e-5, warmup_lr_init=5e-6)
    scheduler = LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=100)
    if not args.qwen2:
        vit = train(args, vit, train_dl, opt, scheduler, epoch=100)
    else:
        vit = train_qwen2(args, vit, train_dl, opt, scheduler, epoch=100)

    print('best_acc:', args.best_acc)
    with open('results_lp.log', 'a') as f:
        f.write(f"dataset {args.dataset}, acc {args.best_acc}, h lr {args.h_lr}, is mm trained {args.mm_trained}\n")
