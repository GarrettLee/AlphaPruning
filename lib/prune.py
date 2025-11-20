import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders
import numpy as np
from collections import defaultdict

import os

from .utils import get_weights, get_modules


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner_model = model.language_model.model
        if hasattr(inner_model, "layers"):
            return inner_model.layers
    raise AttributeError("Unable to locate transformer layers on the provided model.")


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_transformer_layers(model)
        
        
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def prepare_calibration_input(model, dataloader, device):
    layers = get_transformer_layers(model)

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            current_batch = batch if isinstance(batch, dict) else batch[0]
            current_batch = _to_device(current_batch, device)
            if isinstance(current_batch, dict):
                model(**current_batch)
            else:
                model(current_batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    return inps, outs, attention_mask, position_ids

def prepare_calibration_input_opt(model, dataloader, device):
    layers = get_transformer_layers(model)
    
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            current_batch = batch if isinstance(batch, dict) else batch[0]
            current_batch = _to_device(current_batch, device)
            if isinstance(current_batch, dict):
                model(**current_batch)
            else:
                model(current_batch)
        except ValueError:
            pass
        
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    return inps, outs, attention_mask, None


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0):
    blocks = get_transformer_layers(model)
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    print(layerwise_pruning_ratios)
    return layerwise_pruning_ratios

#########################################################################################################################

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    layers = get_transformer_layers(model)
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]

    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            print(ratios[k])
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*ratios[k])].cpu()
            W_mask = (W_metric<=thresh)
            k+=1

            W[W_mask] = 0
    

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None, processor=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer, processor=processor)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    print ("inps",inps)

    layers = get_transformer_layers(model)
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None, processor=None):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer, processor=processor)
    
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = get_transformer_layers(model)
    
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    print('Ready.')
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            print("using !")
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
            k+=1

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

###############################################################################################################
def prune_magnitude_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, processor=None):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # magnitude pruning
    prune_magnitude(args, model, tokenizer, device, ratios=all_layer_ratio)


def prune_wanda_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, processor=None):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio, processor=processor)

def prune_sparsegpt_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, processor=None):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio, processor=processor)