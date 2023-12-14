import os 
import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
# TODO
# import utils.eval_diff as eval_diff
from dataset import dataset_TM_train_diff
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
# from models.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

# TODO
# Define denoiser network HERE
# denoiser = Unet1D

# diffusion = GaussianDiffusion1D(model=denoiser,
#                                 seq_length=args.window_size,
#                                 timesteps=args.timesteps,
#                                 objective=args.obj
#                                 )

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    # diffusion.load_state_dict(ckpt['diff'], strict=True)

trans_encoder.train()
# diffusion.train()
trans_encoder.cuda()
# diffusion.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

##### ---- get code ---- #####
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    file_path = pjoin(args.vq_dir, name[0] +'.npy')
    if not os.path.exists(file_path):
        pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
        target = net.encode(pose)
        target = target.cpu().numpy()
        np.save(file_path, target)

train_loader = dataset_TM_train_diff.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train_diff.cycle(train_loader)

        
##### ---- Training ---- #####
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_diff.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, diffusion, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False)
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False)
while nb_iter <= args.total_iter:
    
    batch = next(train_loader_iter)
    pose, clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs, seq = pose.shape[:2]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
    
    text = clip.tokenize(clip_text, truncate=True).cuda()
    
    feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:,:-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text)
    cls_pred = cls_pred.contiguous()

    pred_stack = torch.zeros((bs, seq, pose.shape[-1])).cuda()
    pred_len = torch.ones(bs).long()
    
    loss_cls = 0.0
    loss_diff = 0.0
    
    for i in range(bs):
        # loss function     (26), (26, 513)
        if i==55:
            print("AYYYYYYY")
            pass
        loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

        # Accuracy
        probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()
        
        pred_pose = net.forward_decoder(cls_pred_index)
        cur_len = pred_pose.shape[1]

        pred_len[i] = min(cur_len, seq)
        pred_stack[i:i+1, :cur_len] = pred_pose[:, :seq]
    
    ## global loss
    loss_total = loss_cls + loss_diff
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    avg_loss_diff = avg_loss_diff + loss_diff.item()
    arg_loss_total = avg_loss_total + loss_total.item()
    nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter == 0:
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_loss_diff = avg_loss_diff / args.print_iter
        avg_loss_total = avg_loss_total / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/cls', avg_loss_cls, nb_iter)
        writer.add_scalar('./Loss/diff', avg_loss_diff, nb_iter)
        writer.add_scalar('./Loss/total', avg_loss_total, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss_cls. {avg_loss_cls:.5f}, Loss_diff. {avg_loss_diff:.5f}, Loss_total. {avg_loss_total:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        avg_loss_diff = 0.
        avg_loss_total = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter == 0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False)
        # best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_diff.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, diffusion, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            