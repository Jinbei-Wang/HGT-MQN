#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict entity-held-out cold-start training for HGTSimGTQueryCLModel.

核心区别：
1) drug_cold：每折 cold drugs 会从训练 heterograph 中物理移除；
              drug-drug similarity graph 也只保留 seen drugs。
2) disease_cold：每折 cold diseases 会从训练 heterograph 中物理移除；
                 disease-disease similarity graph 也只保留 seen diseases。
3) 训练 pair 全部使用 seen-only 子图中的 local id。
4) 测试时 cold entity 不进入模型传播，而是：
      cold raw feature / raw similarity
      -> 检索 top-k seen entities
      -> 聚合 seen entities 的 final_node_emb
      -> pair_decoder 预测 all candidates。

注意：
- 这是严格 unseen-entity cold-start，比“只删除 cold entity 的 drug-disease 边”更严格。
- 该脚本默认每个 epoch 在 all-candidates 冷启动测试集上评估 AUC/AUPR。
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import KFold
import dgl

from load_data2 import load, prepare_similarity_graphs
from utiles.utils import (
    get_metrics_auc,
    get_metrics,
    plot_result_auc,
    plot_result_aupr,
    set_seed,
)
from model_update5 import HGTSimGTQueryCLModel


# ============================================================
# Basic helpers
# ============================================================

def build_label_matrix(dataset: str) -> np.ndarray:
    return pd.read_csv(f"./dataset/{dataset}/{dataset}_baseline.csv", header=None).values.astype(np.int64)


def build_all_pairs(df: np.ndarray):
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])], dtype=np.int64)
    return data, data[data[:, 2] == 1], data[data[:, 2] == 0]


def sample_train_negatives(train_neg_id: np.ndarray, train_pos_n: int, neg_ratio: int, seed: int):
    need_neg = int(neg_ratio) * int(train_pos_n)
    if need_neg > len(train_neg_id):
        raise ValueError(f"Not enough training negatives. Need {need_neg}, only {len(train_neg_id)} available.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_neg_id))
    return train_neg_id[perm[:need_neg]]


def build_pair_tensors(pos_local: np.ndarray, neg_local: np.ndarray, device: th.device):
    pos_pairs = pos_local[:, :2]
    neg_pairs = neg_local[:, :2]
    all_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
    labels = np.concatenate([
        np.ones(len(pos_pairs), dtype=np.float32),
        np.zeros(len(neg_pairs), dtype=np.float32),
    ])
    drug_idx = th.tensor(all_pairs[:, 0], dtype=th.long, device=device)
    disease_idx = th.tensor(all_pairs[:, 1], dtype=th.long, device=device)
    labels = th.tensor(labels, dtype=th.float32, device=device)
    return drug_idx, disease_idx, labels


def safe_load_heterograph(
    dataset: str,
    device: th.device,
    base_dir: Optional[str] = None,
    device_id: Optional[int] = None,
    feature_mode: str = "llm",
):
    try:
        return load(dataset, base_dir=base_dir, device=device, device_id=device_id, feature_mode=feature_mode)
    except TypeError:
        return load(dataset)


def get_feature_dict(g, dataset_name: str) -> Dict[str, th.Tensor]:
    if dataset_name == "Kdataset":
        return {
            "drug": g.nodes["drug"].data["h"],
            "disease": g.nodes["disease"].data["h"],
            "protein": g.nodes["protein"].data["h"],
            "gene": g.nodes["gene"].data["h"],
            "pathway": g.nodes["pathway"].data["h"],
        }
    if dataset_name in ["Bdataset", "Cdataset", "Rdataset"]:
        return {
            "drug": g.nodes["drug"].data["h"],
            "disease": g.nodes["disease"].data["h"],
            "protein": g.nodes["protein"].data["h"],
        }
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def build_binary_criterion(pos_weight: th.Tensor):
    return th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@dataclass
class BestState:
    score: float = -1.0
    epoch: int = 0
    state: Optional[dict] = None
    test_auc: float = -1.0
    test_aupr: float = -1.0
    test_rec: float = -1.0


class MetricTracker:
    def __init__(self):
        self.best = BestState()

    def step(self, monitor: float, epoch: int, state_dict, test_auc: float, test_aupr: float, test_rec: float):
        if monitor > self.best.score:
            self.best.score = float(monitor)
            self.best.epoch = int(epoch)
            self.best.state = copy.deepcopy(state_dict)
            self.best.test_auc = float(test_auc)
            self.best.test_aupr = float(test_aupr)
            self.best.test_rec = float(test_rec)


@th.no_grad()
def evaluate_probs(labels_tensor: th.Tensor, probs_tensor: th.Tensor):
    labels_np = labels_tensor.detach().cpu().numpy()
    probs_np = probs_tensor.detach().cpu().numpy()
    auc, aupr = get_metrics_auc(labels_np, probs_np)
    _, _, acc, f1, pre, rec, spe = get_metrics(labels_np, probs_np)
    return auc, aupr, acc, f1, pre, rec, spe


# ============================================================
# Strict cold-start splitting and local id mapping
# ============================================================

def build_strict_entity_folds(
    df: np.ndarray,
    data_pos: np.ndarray,
    data_neg: np.ndarray,
    split_mode: str,
    n_splits: int,
    seed: int,
):
    """
    Returns folds with original ids.
    drug_cold:
        cold_drugs are held out.
        train_pos/neg only include seen_drugs x all_diseases.
        test_pos/neg include cold_drugs x all_diseases.
    disease_cold:
        cold_diseases are held out.
        train_pos/neg only include all_drugs x seen_diseases.
        test_pos/neg include all_drugs x cold_diseases.
    """
    n_drug, n_dis = df.shape

    if split_mode == "drug_cold":
        entities = np.unique(data_pos[:, 0]).astype(np.int64)
    elif split_mode == "disease_cold":
        entities = np.unique(data_pos[:, 1]).astype(np.int64)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_id, (seen_entity_idx, cold_entity_idx) in enumerate(kf.split(entities), start=1):
        cold_entities = np.sort(entities[cold_entity_idx]).astype(np.int64)

        if split_mode == "drug_cold":
            cold_set = set(cold_entities.tolist())
            seen_drugs = np.array([i for i in range(n_drug) if i not in cold_set], dtype=np.int64)
            seen_diseases = np.arange(n_dis, dtype=np.int64)
            cold_drugs = cold_entities
            cold_diseases = np.array([], dtype=np.int64)

            train_pos_mask = np.array([int(x[0]) not in cold_set for x in data_pos], dtype=bool)
            test_pos_mask = np.array([int(x[0]) in cold_set for x in data_pos], dtype=bool)
            train_neg_mask = np.array([int(x[0]) not in cold_set for x in data_neg], dtype=bool)
            test_neg_mask = np.array([int(x[0]) in cold_set for x in data_neg], dtype=bool)

        else:
            cold_set = set(cold_entities.tolist())
            seen_drugs = np.arange(n_drug, dtype=np.int64)
            seen_diseases = np.array([j for j in range(n_dis) if j not in cold_set], dtype=np.int64)
            cold_drugs = np.array([], dtype=np.int64)
            cold_diseases = cold_entities

            train_pos_mask = np.array([int(x[1]) not in cold_set for x in data_pos], dtype=bool)
            test_pos_mask = np.array([int(x[1]) in cold_set for x in data_pos], dtype=bool)
            train_neg_mask = np.array([int(x[1]) not in cold_set for x in data_neg], dtype=bool)
            test_neg_mask = np.array([int(x[1]) in cold_set for x in data_neg], dtype=bool)

        folds.append({
            "fold": fold_id,
            "split_mode": split_mode,
            "seen_drugs": seen_drugs,
            "seen_diseases": seen_diseases,
            "cold_drugs": cold_drugs,
            "cold_diseases": cold_diseases,
            "train_pos_orig": data_pos[train_pos_mask],
            "test_pos_orig": data_pos[test_pos_mask],
            "train_neg_orig": data_neg[train_neg_mask],
            "test_neg_orig": data_neg[test_neg_mask],
        })

    return folds


def make_orig_to_local(orig_ids: np.ndarray, total_n: int) -> np.ndarray:
    mapping = np.full(total_n, -1, dtype=np.int64)
    mapping[orig_ids] = np.arange(len(orig_ids), dtype=np.int64)
    return mapping


def pairs_orig_to_local(pairs: np.ndarray, drug_map: np.ndarray, disease_map: np.ndarray) -> np.ndarray:
    d = drug_map[pairs[:, 0]]
    s = disease_map[pairs[:, 1]]
    ok = (d >= 0) & (s >= 0)
    if not np.all(ok):
        bad = np.sum(~ok)
        raise ValueError(f"{bad} pairs cannot be mapped to local ids. Check seen/cold split.")
    y = pairs[:, 2] if pairs.shape[1] >= 3 else np.ones(len(pairs), dtype=np.int64)
    return np.stack([d, s, y], axis=1).astype(np.int64)


# ============================================================
# Seen-only graph construction
# ============================================================

def build_seen_only_heterograph(g_full, seen_drugs: np.ndarray, seen_diseases: np.ndarray, device: th.device):
    """
    Physically remove held-out drug/disease nodes by node_subgraph.
    Other node types are retained in full.
    """
    node_dict = {}
    for ntype in g_full.ntypes:
        if ntype == "drug":
            node_dict[ntype] = th.tensor(seen_drugs, dtype=th.long, device=g_full.device)
        elif ntype == "disease":
            node_dict[ntype] = th.tensor(seen_diseases, dtype=th.long, device=g_full.device)
        else:
            node_dict[ntype] = th.arange(g_full.num_nodes(ntype), dtype=th.long, device=g_full.device)

    # relabel_nodes=True: local ids become compact [0, n_seen).
    g_seen = dgl.node_subgraph(g_full, node_dict, relabel_nodes=True, store_ids=True)
    return g_seen.to(device)


def make_dummy_similarity_graph(S_sub: th.Tensor):
    """
    model.sim_encoder.set_similarity_from_graphs only needs graph.ndata['sim_feature'].
    A graph with correct num_nodes is enough.
    """
    n = S_sub.shape[0]
    g = dgl.graph(([], []), num_nodes=n, device=S_sub.device)
    g.ndata["sim_feature"] = S_sub.float()
    return g


def get_sim_matrix_from_graph(sim_graph, device: th.device) -> th.Tensor:
    if "sim_feature" not in sim_graph.ndata:
        raise KeyError("similarity graph ndata['sim_feature'] not found")
    return sim_graph.ndata["sim_feature"].float().to(device)


def build_seen_similarity_graphs(
    full_dr_graph,
    full_di_graph,
    seen_drugs: np.ndarray,
    seen_diseases: np.ndarray,
    device: th.device,
):
    S_dr_full = get_sim_matrix_from_graph(full_dr_graph, device)
    S_di_full = get_sim_matrix_from_graph(full_di_graph, device)

    seen_dr_t = th.tensor(seen_drugs, dtype=th.long, device=device)
    seen_di_t = th.tensor(seen_diseases, dtype=th.long, device=device)

    S_dr_seen = S_dr_full.index_select(0, seen_dr_t).index_select(1, seen_dr_t)
    S_di_seen = S_di_full.index_select(0, seen_di_t).index_select(1, seen_di_t)

    return make_dummy_similarity_graph(S_dr_seen), make_dummy_similarity_graph(S_di_seen), S_dr_full, S_di_full


# ============================================================
# Forward / optimizer / train
# ============================================================

def model_forward(model, g_het, x_het, dr_graph, di_graph, drug_idx, disease_idx, return_aux=False):
    return model(
        g_het=g_het,
        x_het=x_het,
        drdr_graph=dr_graph,
        didi_graph=di_graph,
        drug_idx=drug_idx,
        disease_idx=disease_idx,
        return_aux=return_aux,
    )


def build_grouped_optimizer(model, args):
    base_lr = args.learning_rate
    hgt_lr = args.hgt_learning_rate if args.hgt_learning_rate is not None else base_lr
    sim_lr = args.sim_learning_rate if args.sim_learning_rate is not None else base_lr
    query_lr = args.query_learning_rate if args.query_learning_rate is not None else base_lr

    hgt_params = []
    hgt_params += list(model.drug_linear.parameters())
    hgt_params += list(model.disease_linear.parameters())
    hgt_params += list(model.other_linear.parameters())
    hgt_params += list(model.hgt_layers.parameters())

    sim_params = list(model.sim_encoder.parameters())

    query_params = []
    query_params += list(model.query_blocks.parameters())
    query_params += list(model.drug_layer_attn.parameters())
    query_params += list(model.disease_layer_attn.parameters())
    query_params += list(model.sim_drug_layer_attn.parameters())
    query_params += list(model.sim_disease_layer_attn.parameters())
    query_params += list(model.pair_decoder.parameters())

    optimizer = th.optim.Adam(
        [
            {"params": hgt_params, "lr": hgt_lr, "weight_decay": args.weight_decay, "name": "hgt_global"},
            {"params": sim_params, "lr": sim_lr, "weight_decay": args.weight_decay, "name": "sim_gt"},
            {"params": query_params, "lr": query_lr, "weight_decay": args.weight_decay, "name": "query_decoder"},
        ]
    )
    print(
        f"[Optimizer] lr setting | HGT/global={hgt_lr:g}, Sim-GT={sim_lr:g}, "
        f"Query/decoder={query_lr:g}, weight_decay={args.weight_decay:g}"
    )
    return optimizer


def train_one_epoch(
    model,
    g_het,
    x_het,
    dr_graph,
    di_graph,
    drug_idx,
    disease_idx,
    labels,
    optimizer,
    criterion,
    lambda_cl_het: float = 0.0,
    lambda_cl_sim: float = 0.0,
    grad_clip: float = 5.0,
):
    model.train()
    n = labels.shape[0]
    perm = th.randperm(n, device=labels.device)
    drug_idx = drug_idx[perm]
    disease_idx = disease_idx[perm]
    labels = labels[perm]

    optimizer.zero_grad(set_to_none=True)
    logits, aux = model_forward(
        model, g_het, x_het, dr_graph, di_graph,
        drug_idx, disease_idx, return_aux=True,
    )
    pred_loss = criterion(logits, labels)
    cl_total, cl_het, cl_sim = model.contrastive_loss(
        aux,
        lambda_cl_het=lambda_cl_het,
        lambda_cl_sim=lambda_cl_sim,
    )
    loss = pred_loss + cl_total
    loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    probs = th.sigmoid(logits.detach())
    train_auc, train_aupr, *_ = evaluate_probs(labels.detach(), probs.detach())

    def to_float(x, default=0.0):
        if x is None:
            return default
        if th.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    return {
        "loss": float(loss.detach().cpu().item()),
        "pred_loss": float(pred_loss.detach().cpu().item()),
        "cl_total": float(cl_total.detach().cpu().item()) if th.is_tensor(cl_total) else float(cl_total),
        "cl_het": float(cl_het.detach().cpu().item()) if th.is_tensor(cl_het) else float(cl_het),
        "cl_sim": float(cl_sim.detach().cpu().item()) if th.is_tensor(cl_sim) else float(cl_sim),
        "query_gamma": to_float(aux.get("query_gamma_mean", None)),
        "drug_gate": to_float(aux.get("drug_query_gate_mean", None)),
        "disease_gate": to_float(aux.get("disease_query_gate_mean", None)),
        "train_auc": train_auc,
        "train_aupr": train_aupr,
    }


@th.no_grad()
def get_final_node_embeddings(model, g_het, x_het, dr_graph, di_graph, device: th.device):
    """
    Run one forward pass to obtain final_node_emb for all seen drug/disease nodes.
    We feed a dummy pair only to trigger forward; aux contains full final_node_emb.
    """
    model.eval()
    dummy_drug = th.zeros(1, dtype=th.long, device=device)
    dummy_dis = th.zeros(1, dtype=th.long, device=device)
    _, aux = model_forward(
        model, g_het, x_het, dr_graph, di_graph,
        dummy_drug, dummy_dis, return_aux=True,
    )
    return aux["final_node_emb"], aux


# ============================================================
# Pseudo-final cold-start inference
# ============================================================

def minmax_norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def cosine_to_seen(raw_all: th.Tensor, cold_orig: int, seen_orig: np.ndarray, device: th.device) -> np.ndarray:
    raw_all = raw_all.to(device).float()
    cold_vec = raw_all[cold_orig].unsqueeze(0)
    seen_idx_t = th.tensor(seen_orig, dtype=th.long, device=device)
    seen_vec = raw_all.index_select(0, seen_idx_t)
    cold_norm = F.normalize(cold_vec, p=2, dim=1)
    seen_norm = F.normalize(seen_vec, p=2, dim=1)
    sims = th.mm(seen_norm, cold_norm.t()).squeeze(1)
    return sims.detach().cpu().numpy().astype(np.float32)


def graph_sim_to_seen(S_full: th.Tensor, cold_orig: int, seen_orig: np.ndarray, device: th.device) -> np.ndarray:
    seen_idx_t = th.tensor(seen_orig, dtype=th.long, device=device)
    sims = S_full[cold_orig].index_select(0, seen_idx_t)
    return sims.detach().cpu().numpy().astype(np.float32)


def build_retrieval_scores(
    raw_all: th.Tensor,
    S_full: th.Tensor,
    cold_orig: int,
    seen_orig: np.ndarray,
    source: str,
    alpha: float,
    device: th.device,
) -> np.ndarray:
    if len(seen_orig) == 0:
        raise ValueError("No seen entities for retrieval.")

    if source == "raw":
        return minmax_norm_np(cosine_to_seen(raw_all, cold_orig, seen_orig, device))
    if source == "sim":
        return minmax_norm_np(graph_sim_to_seen(S_full, cold_orig, seen_orig, device))
    if source == "mixed":
        raw_s = minmax_norm_np(cosine_to_seen(raw_all, cold_orig, seen_orig, device))
        sim_s = minmax_norm_np(graph_sim_to_seen(S_full, cold_orig, seen_orig, device))
        return float(alpha) * raw_s + (1.0 - float(alpha)) * sim_s
    raise ValueError(f"Unsupported retrieval source: {source}")


def pseudo_final_embedding(
    final_seen_emb: th.Tensor,
    raw_all: th.Tensor,
    S_full: th.Tensor,
    cold_orig: int,
    seen_orig: np.ndarray,
    seen_orig_to_local: np.ndarray,
    k: int,
    temperature: float,
    source: str,
    alpha: float,
    device: th.device,
):
    """
    final_seen_emb: [N_seen, D] local order
    seen_orig: original ids in local order
    """
    scores = build_retrieval_scores(raw_all, S_full, cold_orig, seen_orig, source, alpha, device)
    k_eff = max(1, min(int(k), len(seen_orig)))
    top_pos = np.argsort(-scores)[:k_eff]
    top_orig = seen_orig[top_pos]
    top_local = seen_orig_to_local[top_orig]
    if np.any(top_local < 0):
        raise RuntimeError("Top-k retrieved entity is not mapped to local seen id.")

    top_scores = th.tensor(scores[top_pos], dtype=th.float32, device=device)
    weights = th.softmax(top_scores / float(temperature), dim=0)
    top_local_t = th.tensor(top_local, dtype=th.long, device=device)
    top_emb = final_seen_emb.index_select(0, top_local_t)
    pseudo = th.sum(weights.unsqueeze(-1) * top_emb, dim=0)

    detail = {
        "cold_orig": int(cold_orig),
        "top_orig": [int(x) for x in top_orig.tolist()],
        "scores": [float(x) for x in scores[top_pos].tolist()],
        "weights": [float(x) for x in weights.detach().cpu().numpy().tolist()],
    }
    return pseudo, detail


def decode_drug_to_all_diseases(pair_decoder, drug_h: th.Tensor, disease_h_all: th.Tensor, chunk_size: int = 8192):
    probs = []
    n = disease_h_all.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        dis_h = disease_h_all[start:end]
        drug_exp = drug_h.unsqueeze(0).expand(end - start, -1)
        pair_feat = pair_decoder.pair_feature(drug_exp, dis_h)
        logits = pair_decoder.mlp(pair_feat).squeeze(-1)
        probs.append(th.sigmoid(logits))
    return th.cat(probs, dim=0)


def decode_all_drugs_to_disease(pair_decoder, drug_h_all: th.Tensor, disease_h: th.Tensor, chunk_size: int = 8192):
    probs = []
    n = drug_h_all.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        drug_h = drug_h_all[start:end]
        dis_exp = disease_h.unsqueeze(0).expand(end - start, -1)
        pair_feat = pair_decoder.pair_feature(drug_h, dis_exp)
        logits = pair_decoder.mlp(pair_feat).squeeze(-1)
        probs.append(th.sigmoid(logits))
    return th.cat(probs, dim=0)


@th.no_grad()
def cold_start_evaluate_all_candidates(
    model,
    final_node_emb: Dict[str, th.Tensor],
    df: np.ndarray,
    split_mode: str,
    seen_drugs: np.ndarray,
    seen_diseases: np.ndarray,
    cold_drugs: np.ndarray,
    cold_diseases: np.ndarray,
    drug_orig_to_local: np.ndarray,
    disease_orig_to_local: np.ndarray,
    raw_drug_all: th.Tensor,
    raw_disease_all: th.Tensor,
    S_dr_full: th.Tensor,
    S_di_full: th.Tensor,
    args,
    device: th.device,
    return_predictions: bool = False,
    return_pseudo_details: bool = False,
):
    model.eval()
    labels_list = []
    probs_list = []
    pred_rows = []
    pseudo_details = []

    if split_mode == "drug_cold":
        disease_h_all = final_node_emb["disease"]  # all diseases are seen in drug-cold
        for cold_d in cold_drugs:
            pseudo_h, detail = pseudo_final_embedding(
                final_seen_emb=final_node_emb["drug"],
                raw_all=raw_drug_all,
                S_full=S_dr_full,
                cold_orig=int(cold_d),
                seen_orig=seen_drugs,
                seen_orig_to_local=drug_orig_to_local,
                k=args.cold_pseudo_k,
                temperature=args.cold_pseudo_temperature,
                source=args.cold_retrieval_source,
                alpha=args.cold_retrieval_alpha,
                device=device,
            )
            probs = decode_drug_to_all_diseases(model.pair_decoder, pseudo_h, disease_h_all, args.eval_chunk_size)
            labels = th.tensor(df[int(cold_d), seen_diseases], dtype=th.float32, device=device)

            labels_list.append(labels)
            probs_list.append(probs)

            if return_predictions:
                for local_j, orig_j in enumerate(seen_diseases):
                    pred_rows.append((int(cold_d), int(orig_j), int(labels[local_j].item()), float(probs[local_j].item())))
            if return_pseudo_details:
                pseudo_details.append(detail)

    elif split_mode == "disease_cold":
        drug_h_all = final_node_emb["drug"]  # all drugs are seen in disease-cold
        for cold_s in cold_diseases:
            pseudo_h, detail = pseudo_final_embedding(
                final_seen_emb=final_node_emb["disease"],
                raw_all=raw_disease_all,
                S_full=S_di_full,
                cold_orig=int(cold_s),
                seen_orig=seen_diseases,
                seen_orig_to_local=disease_orig_to_local,
                k=args.cold_pseudo_k,
                temperature=args.cold_pseudo_temperature,
                source=args.cold_retrieval_source,
                alpha=args.cold_retrieval_alpha,
                device=device,
            )
            probs = decode_all_drugs_to_disease(model.pair_decoder, drug_h_all, pseudo_h, args.eval_chunk_size)
            labels = th.tensor(df[seen_drugs, int(cold_s)], dtype=th.float32, device=device)

            labels_list.append(labels)
            probs_list.append(probs)

            if return_predictions:
                for local_i, orig_i in enumerate(seen_drugs):
                    pred_rows.append((int(orig_i), int(cold_s), int(labels[local_i].item()), float(probs[local_i].item())))
            if return_pseudo_details:
                pseudo_details.append(detail)
    else:
        raise ValueError(split_mode)

    labels_all = th.cat(labels_list, dim=0)
    probs_all = th.cat(probs_list, dim=0)
    metrics = evaluate_probs(labels_all, probs_all)
    return metrics, labels_all, probs_all, pred_rows, pseudo_details


# ============================================================
# Args
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-da", "--dataset", default="kdataset")
    parser.add_argument("-id", "--device_id", default="0")
    parser.add_argument("-fo", "--nfold", type=int, default=10)
    parser.add_argument("-nr", "--neg_ratio", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("-sp", "--saved_path", default="result_strict_cold")
    parser.add_argument("--feature_mode", choices=["llm", "random"], default="llm")
    parser.add_argument("--save_plots", action="store_true", default=True)

    parser.add_argument(
        "--split_mode",
        choices=["drug_cold", "disease_cold"],
        required=True,
        help="Strict entity-held-out mode. Held-out entities are physically removed from training graph."
    )

    # training
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--bce_pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.5)

    # model
    parser.add_argument("--hidden_feats", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_sim_layers", type=int, default=2)
    parser.add_argument("--query_layers", type=str, default="1")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pair_hidden", type=int, default=128)
    parser.add_argument("--pair_mode", choices=["rotate", "absdiff"], default="rotate")
    parser.add_argument("--layer_pooling", choices=["last", "mean", "dream", "attn"], default="attn")

    parser.add_argument("--hgt_learning_rate", type=float, default=None)
    parser.add_argument("--sim_learning_rate", type=float, default=1e-3)
    parser.add_argument("--query_learning_rate", type=float, default=None)

    # similarity / query
    parser.add_argument("--sim_topk", type=int, default=5)
    parser.add_argument("--sim_use_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_no_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_diffusion_alpha", type=float, default=0.15)
    parser.add_argument("--sim_diffusion_steps", type=int, default=3)
    parser.add_argument("--sim_use_diffused_adj_for_gcn", action="store_true", default=False)
    parser.add_argument("--query_gamma_init", type=float, default=0.05)

    # contrastive learning
    parser.add_argument("--lambda_cl_het", type=float, default=0.1,
                        help="For strict cold-start, 0 is safest unless CL only uses seen nodes.")
    parser.add_argument("--lambda_cl_sim", type=float, default=0.0)
    parser.add_argument("--cl_temperature", type=float, default=0.2)
    parser.add_argument("--cl_sample_size", type=int, default=0)

    # strict cold-start pseudo-final inference
    parser.add_argument("--cold_pseudo_k", type=int, default=10)
    parser.add_argument("--cold_pseudo_temperature", type=float, default=0.1)
    parser.add_argument("--cold_retrieval_source", choices=["raw", "sim", "mixed"], default="raw")
    parser.add_argument("--cold_retrieval_alpha", type=float, default=0.7,
                        help="Only for mixed retrieval: alpha*raw_similarity + (1-alpha)*sim_similarity.")
    parser.add_argument("--eval_chunk_size", type=int, default=8192)
    parser.add_argument("--save_fold_predictions", action="store_true", default=False)
    parser.add_argument("--save_pseudo_details", action="store_true", default=False)

    parser.add_argument("--resume_skip_finished", action="store_true", default=True,
                        help="If a fold already has best_model_state.pth and fold_metrics.json, skip it.")
    parser.add_argument("--force_rerun_finished", action="store_true", default=False,
                        help="Ignore existing checkpoints and rerun all folds.")

    return parser


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()
    if args.sim_no_diffusion:
        args.sim_use_diffusion = False

    set_seed(args.seed)
    os.makedirs(args.saved_path, exist_ok=True)

    log_file = open(os.path.join(args.saved_path, "train.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    print(args)
    device = th.device(f"cuda:{args.device_id}" if th.cuda.is_available() else "cpu")
    print("Training on", device)

    base_dir = args.base_dir or os.path.join("./dataset", args.dataset)
    df = build_label_matrix(args.dataset)
    data, data_pos, data_neg = build_all_pairs(df)

    n_drug_full, n_dis_full = df.shape
    print(f"Matrix shape: {n_drug_full} x {n_dis_full}")
    print(f"Positive samples: {len(data_pos):,}")
    print(f"Negative samples: {len(data_neg):,}")
    print(f"Total pairs: {len(data):,}")
    print(f"Strict split mode: {args.split_mode}")

    # Load full heterograph once. Each fold then induces seen-only node subgraph.
    g_full = safe_load_heterograph(
        args.dataset,
        device=device,
        base_dir=base_dir,
        device_id=int(args.device_id),
        feature_mode=args.feature_mode,
    ).to(device)

    # Raw features from full graph are used only for cold-entity retrieval at inference.
    x_full = get_feature_dict(g_full, args.dataset)
    raw_drug_all = x_full["drug"].detach().to(device)
    raw_disease_all = x_full["disease"].detach().to(device)

    # Full similarity matrices are used:
    # - fold training uses seen-only submatrix.
    # - cold retrieval uses cold-vs-seen similarities.
    full_dr_graph, full_di_graph = prepare_similarity_graphs(
        dataset=args.dataset,
        base_dir=base_dir,
        K=args.sim_topk,
        device=device,
        make_undirected=True,
    )
    print(f"[Full Similarity Graphs] n_drug={full_dr_graph.num_nodes()}, n_dis={full_di_graph.num_nodes()}")

    folds = build_strict_entity_folds(
        df=df,
        data_pos=data_pos,
        data_neg=data_neg,
        split_mode=args.split_mode,
        n_splits=args.nfold,
        seed=args.seed,
    )

    pred_result = np.zeros(df.shape, dtype=np.float32)
    pred_mask = np.zeros(df.shape, dtype=bool)
    fold_metrics = []

    for fold_data in folds:
        fold = fold_data["fold"]

        fold_dir = os.path.join(args.saved_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        best_ckpt_path = os.path.join(fold_dir, "best_model_state.pth")

        print("\n" + "=" * 80)
        print(f"Strict {args.split_mode} | {args.nfold}-Fold CV | Fold {fold}")
        print("=" * 80)

        print(f"[Resume check] fold_dir = {fold_dir}")
        print(f"[Resume check] best_ckpt_path = {best_ckpt_path}")
        print(f"[Resume check] checkpoint exists = {os.path.isfile(best_ckpt_path)}")

        if os.path.isfile(best_ckpt_path):
            print(f"[Resume] Fold {fold} already has best_model_state.pth. Skip training this fold.")
            continue
        print("\n" + "=" * 80)
        print(f"Strict {args.split_mode} | {args.nfold}-Fold CV | Fold {fold}")
        print("=" * 80)

        seen_drugs = fold_data["seen_drugs"]
        seen_diseases = fold_data["seen_diseases"]
        cold_drugs = fold_data["cold_drugs"]
        cold_diseases = fold_data["cold_diseases"]

        print(f"#Seen drugs:    {len(seen_drugs):,} / {n_drug_full:,}")
        print(f"#Seen diseases: {len(seen_diseases):,} / {n_dis_full:,}")
        print(f"#Cold drugs:    {len(cold_drugs):,}")
        print(f"#Cold diseases: {len(cold_diseases):,}")

        train_pos_orig = fold_data["train_pos_orig"]
        train_neg_orig = fold_data["train_neg_orig"]
        test_pos_orig = fold_data["test_pos_orig"]
        test_neg_orig = fold_data["test_neg_orig"]

        print(f"Train pos pool: {len(train_pos_orig):,}")
        print(f"Train neg pool: {len(train_neg_orig):,}")
        print(f"Test pos:       {len(test_pos_orig):,}")
        print(f"Test neg(all):  {len(test_neg_orig):,}")

        # local id maps
        drug_orig_to_local = make_orig_to_local(seen_drugs, n_drug_full)
        disease_orig_to_local = make_orig_to_local(seen_diseases, n_dis_full)

        train_pos_local = pairs_orig_to_local(train_pos_orig, drug_orig_to_local, disease_orig_to_local)
        train_neg_local_pool = pairs_orig_to_local(train_neg_orig, drug_orig_to_local, disease_orig_to_local)

        sampled_train_neg_local = sample_train_negatives(
            train_neg_local_pool[:, :2],
            len(train_pos_local),
            args.neg_ratio,
            args.seed + fold * 20000,
        )
        print(f"Sampled train neg: {len(sampled_train_neg_local):,} (neg_ratio={args.neg_ratio})")

        # build seen-only training heterograph and similarity graphs
        g_seen = build_seen_only_heterograph(g_full, seen_drugs, seen_diseases, device)
        x_seen = {k: v.to(device) for k, v in get_feature_dict(g_seen, args.dataset).items()}

        dr_seen_graph, di_seen_graph, S_dr_full, S_di_full = build_seen_similarity_graphs(
            full_dr_graph, full_di_graph, seen_drugs, seen_diseases, device
        )

        n_drug_seen = dr_seen_graph.num_nodes()
        n_dis_seen = di_seen_graph.num_nodes()
        print(f"[Seen graph] drug={g_seen.num_nodes('drug')}, disease={g_seen.num_nodes('disease')}")
        print(f"[Seen sim] n_drug={n_drug_seen}, n_dis={n_dis_seen}, sim_topk={args.sim_topk}")

        model = HGTSimGTQueryCLModel(
            etypes=g_seen.etypes,
            ntypes=g_seen.ntypes,
            n_drug=n_drug_seen,
            n_dis=n_dis_seen,
            in_feats=args.hidden_feats,
            hidden_feats=args.hidden_feats,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_hgt_layers=args.num_hgt_layers,
            num_sim_layers=args.num_sim_layers,
            query_layers=args.query_layers,
            pair_hidden=args.pair_hidden,
            pair_mode=args.pair_mode,
            sim_topk=args.sim_topk,
            sim_use_diffusion=args.sim_use_diffusion,
            sim_diffusion_alpha=args.sim_diffusion_alpha,
            sim_diffusion_steps=args.sim_diffusion_steps,
            sim_use_diffused_adj_for_gcn=args.sim_use_diffused_adj_for_gcn,
            query_gamma_init=args.query_gamma_init,
            layer_pooling=args.layer_pooling,
            cl_temperature=args.cl_temperature,
            cl_sample_size=args.cl_sample_size,
        ).to(device)
        model.set_similarity_graphs(dr_seen_graph, di_seen_graph, device=device)
        print("[Strict cold-start] Model sees only seen entities in heterograph and similarity graph.")

        train_drug_idx, train_disease_idx, train_labels = build_pair_tensors(
            train_pos_local[:, :2],
            sampled_train_neg_local,
            device,
        )

        pos_weight = th.tensor(
            (len(sampled_train_neg_local) / max(1, len(train_pos_local))) * args.bce_pos_weight_scale,
            dtype=th.float32,
            device=device,
        )
        criterion = build_binary_criterion(pos_weight)
        optimizer = build_grouped_optimizer(model, args)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=args.patience, factor=0.5
        )
        tracker = MetricTracker()

        # save mapping/metadata for audit and standalone evaluation
        metadata = {
            "fold": fold,
            "split_mode": args.split_mode,
            "seen_drugs": seen_drugs.tolist(),
            "seen_diseases": seen_diseases.tolist(),
            "cold_drugs": cold_drugs.tolist(),
            "cold_diseases": cold_diseases.tolist(),
            "neg_ratio": args.neg_ratio,
            "cold_pseudo_k": args.cold_pseudo_k,
            "cold_pseudo_temperature": args.cold_pseudo_temperature,
            "cold_retrieval_source": args.cold_retrieval_source,
            "cold_retrieval_alpha": args.cold_retrieval_alpha,
        }
        with open(os.path.join(fold_dir, "strict_cold_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        t_train0 = time.time()
        for epoch in range(1, args.epoch + 1):
            train_stats = train_one_epoch(
                model=model,
                g_het=g_seen,
                x_het=x_seen,
                dr_graph=dr_seen_graph,
                di_graph=di_seen_graph,
                drug_idx=train_drug_idx,
                disease_idx=train_disease_idx,
                labels=train_labels,
                optimizer=optimizer,
                criterion=criterion,
                lambda_cl_het=args.lambda_cl_het,
                lambda_cl_sim=args.lambda_cl_sim,
                grad_clip=args.grad_clip,
            )

            # Cold-start all-candidate evaluation
            final_node_emb, _ = get_final_node_embeddings(model, g_seen, x_seen, dr_seen_graph, di_seen_graph, device)
            metrics, labels_all, probs_all, _, _ = cold_start_evaluate_all_candidates(
                model=model,
                final_node_emb=final_node_emb,
                df=df,
                split_mode=args.split_mode,
                seen_drugs=seen_drugs,
                seen_diseases=seen_diseases,
                cold_drugs=cold_drugs,
                cold_diseases=cold_diseases,
                drug_orig_to_local=drug_orig_to_local,
                disease_orig_to_local=disease_orig_to_local,
                raw_drug_all=raw_drug_all,
                raw_disease_all=raw_disease_all,
                S_dr_full=S_dr_full,
                S_di_full=S_di_full,
                args=args,
                device=device,
                return_predictions=False,
                return_pseudo_details=False,
            )
            test_auc, test_aupr, _, test_f1, test_pre, test_rec, test_spe = metrics

            scheduler.step(test_aupr)
            tracker.step(test_aupr, epoch, model.state_dict(), test_auc, test_aupr, test_rec)

            t_now = time.time()
            train_time = (t_now - t_train0) / 60.0

            if epoch == 1 or epoch == args.epoch or epoch % 10 == 0:
                print(
                    f"[fold {fold}] Epoch {epoch:03d} | "
                    f"Loss {train_stats['loss']:.4f} | Pred {train_stats['pred_loss']:.4f} | "
                    f"CLhet {train_stats['cl_het']:.4f}*{args.lambda_cl_het:g} | "
                    f"CLsim {train_stats['cl_sim']:.4f}*{args.lambda_cl_sim:g} | "
                    f"QGamma {train_stats['query_gamma']:.4f} | "
                    f"GateD/S {train_stats['drug_gate']:.3f}/{train_stats['disease_gate']:.3f} | "
                    f"Train AUPR {train_stats['train_aupr']:.4f} | "
                    f"|| Cold Test AUC {test_auc:.4f} | Cold Test AUPR {test_aupr:.4f} | "
                    f"Cold F1 {test_f1:.4f} | Cold Recall {test_rec:.4f} | Cold Pre {test_pre:.4f} | "
                    f"Time {train_time:.2f}min"
                )

        if tracker.best.state is not None:
            model.load_state_dict(tracker.best.state)
            th.save(tracker.best.state, os.path.join(fold_dir, "best_model_state.pth"))
            print(
                f"[fold {fold}] Best checkpoint | Epoch {tracker.best.epoch} | "
                f"Cold Test AUC {tracker.best.test_auc:.4f} | "
                f"Cold Test AUPR {tracker.best.test_aupr:.4f} | Rec {tracker.best.test_rec:.4f}"
            )

        # Final fold evaluation with best checkpoint
        final_node_emb, _ = get_final_node_embeddings(model, g_seen, x_seen, dr_seen_graph, di_seen_graph, device)
        metrics, labels_all, probs_all, pred_rows, pseudo_details = cold_start_evaluate_all_candidates(
            model=model,
            final_node_emb=final_node_emb,
            df=df,
            split_mode=args.split_mode,
            seen_drugs=seen_drugs,
            seen_diseases=seen_diseases,
            cold_drugs=cold_drugs,
            cold_diseases=cold_diseases,
            drug_orig_to_local=drug_orig_to_local,
            disease_orig_to_local=disease_orig_to_local,
            raw_drug_all=raw_drug_all,
            raw_disease_all=raw_disease_all,
            S_dr_full=S_dr_full,
            S_di_full=S_di_full,
            args=args,
            device=device,
            return_predictions=args.save_fold_predictions,
            return_pseudo_details=args.save_pseudo_details,
        )
        fold_auc, fold_aupr, fold_acc, fold_f1, fold_pre, fold_rec, fold_spe = metrics

        print(f"\nFold {fold} Final Strict Cold AUC:      {fold_auc:.4f}")
        print(f"Fold {fold} Final Strict Cold AUPR:     {fold_aupr:.4f}")
        print(f"Fold {fold} Final Strict Cold ACC:      {fold_acc:.4f}")
        print(f"Fold {fold} Final Strict Cold F1:       {fold_f1:.4f}")
        print(f"Fold {fold} Final Strict Cold Rec:      {fold_rec:.4f}")
        print(f"Fold {fold} Final Strict Cold Pre:      {fold_pre:.4f}")
        print(f"Fold {fold} Final Strict Cold Spe:      {fold_spe:.4f}")

        fold_metric = {
            "fold": int(fold),
            "auc": float(fold_auc),
            "aupr": float(fold_aupr),
            "acc": float(fold_acc),
            "f1": float(fold_f1),
            "pre": float(fold_pre),
            "rec": float(fold_rec),
            "spe": float(fold_spe),
        }

        fold_metrics.append(fold_metric)

        with open(os.path.join(fold_dir, "fold_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(fold_metric, f, indent=2, ensure_ascii=False)

        if args.save_fold_predictions:
            pred_path = os.path.join(fold_dir, "cold_predictions.csv")
            pd.DataFrame(pred_rows, columns=["drug_id", "disease_id", "label", "score"]).to_csv(pred_path, index=False)
            print(f"Saved fold predictions: {pred_path}")

        if args.save_pseudo_details:
            detail_path = os.path.join(fold_dir, "pseudo_details.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(pseudo_details, f, indent=2, ensure_ascii=False)
            print(f"Saved pseudo details: {detail_path}")

        # Fill global prediction matrix for cold test region.
        if args.split_mode == "drug_cold":
            idx = 0
            for cold_d in cold_drugs:
                n = len(seen_diseases)
                scores = probs_all[idx:idx+n].detach().cpu().numpy()
                pred_result[int(cold_d), seen_diseases] = scores
                pred_mask[int(cold_d), seen_diseases] = True
                idx += n
        else:
            idx = 0
            for cold_s in cold_diseases:
                n = len(seen_drugs)
                scores = probs_all[idx:idx+n].detach().cpu().numpy()
                pred_result[seen_drugs, int(cold_s)] = scores
                pred_mask[seen_drugs, int(cold_s)] = True
                idx += n

    # Overall metrics only over evaluated cold regions.
    overall_labels = df[pred_mask].reshape(-1)
    overall_preds = pred_result[pred_mask].reshape(-1)
    overall_auc, overall_aupr = get_metrics_auc(overall_labels, overall_preds)
    _, _, overall_acc, overall_f1, overall_pre, overall_rec, overall_spe = get_metrics(overall_labels, overall_preds)

    print("\n" + "=" * 80)
    print("Overall Strict Cold-Start Results")
    print("=" * 80)
    print(f"Evaluated pairs: {pred_mask.sum():,}")
    print(f"Overall AUC:  {overall_auc:.4f}")
    print(f"Overall AUPR: {overall_aupr:.4f}")
    print(f"Overall Acc:  {overall_acc:.4f}")
    print(f"Overall F1:   {overall_f1:.4f}")
    print(f"Overall Rec:  {overall_rec:.4f}")
    print(f"Overall Pre:  {overall_pre:.4f}")
    print(f"Overall Spe:  {overall_spe:.4f}")

    if len(fold_metrics) > 0:
        print("\nPer-fold Mean ± SD")
        for key in ["auc", "aupr", "acc", "f1", "pre", "rec", "spe"]:
            vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
            print(f"{key.upper()}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

        summary_path = os.path.join(args.saved_path, f"summary_{args.split_mode}.csv")
        rows = []
        for m in fold_metrics:
            rows.append(m)
        mean_row = {"fold": "mean"}
        std_row = {"fold": "std"}
        for key in ["auc", "aupr", "acc", "f1", "pre", "rec", "spe"]:
            vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
            mean_row[key] = vals.mean()
            std_row[key] = vals.std(ddof=1)
        rows.extend([mean_row, std_row])
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

    if args.save_plots:
        plot_result_auc(args, overall_labels, overall_preds, overall_auc)
        plot_result_aupr(args, overall_labels, overall_preds, overall_aupr)
        print("Saved overall ROC curve:", os.path.join(args.saved_path, "result_auc.png"))
        print("Saved overall PR curve:", os.path.join(args.saved_path, "result_aupr.png"))

    log_file.close()


if __name__ == "__main__":
    main()
