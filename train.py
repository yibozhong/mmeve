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
from transformers import CLIPProcessor, CLIPModel, Qwen2VLForConditionalGeneration, AutoProcessor
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

def get_dfn_vision_model():
    print(f"getting dfn vision model")
    model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
    return model.visual

def get_qwen2vl_vision_encoder():
    print(f"getting qwen2vl vision encoder")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    with open('qwenconfig', 'a') as f:
        f.write(str(model.visual.config))
    import copy
    backbone = copy.deepcopy(model.visual)
    
    # 3. 显式释放原模型
    del model
    torch.cuda.empty_cache()  # 清理GPU缓存
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return backbone


def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            opt.zero_grad()
            # 更新 autocast 的用法
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

def train_qwen(args, model, dl, opt, scheduler, epoch):
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
    parser.add_argument('--qwen', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    args.best_acc = 0

    # define the model and dataloader
    if not args.mm_trained:
        backbone = get_dfn_vision_model()
    else:
        backbone = get_qwen2vl_vision_encoder()
    # backbone.config._attn_implementation = 'flash_attention_2'
    # print(backbone.config._attn_implementation)
    if not args.qwen:
        vit = ViTLargeInitFromOpenAICliP(backbone, get_classes_num(args.dataset))
        train_dl, test_dl = get_data(args.dataset, normalize=False)
    else:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        vit = ViTInitFromQwenVL(backbone, get_classes_num(args.dataset))
        train_dl, test_dl = get_data_for_qwen(args.dataset, normalize=False)
    
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
    if not args.qwen:
        vit = train(args, vit, train_dl, opt, scheduler, epoch=100)
    else:
        vit = train_qwen(args, vit, train_dl, opt, scheduler, epoch=100)

    print('best_acc:', args.best_acc)
    with open('results_lp.log', 'a') as f:
        f.write(f"dataset {args.dataset}, acc {args.best_acc}, h lr {args.h_lr}, is mm trained {args.mm_trained}\n")
