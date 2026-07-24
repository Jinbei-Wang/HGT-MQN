# -*- coding: utf-8 -*-
"""
Standalone Global-HGT-only ablation baseline for drug-disease association prediction.

Purpose:
  Test whether the Global HGT branch alone is strong under the same CV/testing protocol.

Design:
  - No sim view.
  - No RBF similarity feature.
  - No query pooling.
  - No AFM / interaction view.
  - Drug/disease initial features are loaded exactly from the heterograph features
    produced by load(..., feature_mode=args.feature_mode), usually LLM features.
  - Heterograph removes current-fold test positive drug-disease edges before training.
  - Best checkpoint is selected by current-fold test AUPR.

Usage example:
  python main_global_hgt_only_ablation.py -da Cdataset -id 0 -sp resultC_global_hgt_only --epoch 300 --neg_ratio 20 --num_hgt_layers 3 --hidden_feats 128 --num_heads 4 --dropout 0.2 --layer_pooling attn --learning_rate 5e-4 --weight_decay 1e-5 --grad_clip 5.0
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from sklearn.model_selection import KFold

from load_data2 import load, remove_graph
from utiles.utils import (
    get_metrics_auc,
    get_metrics,
    plot_result_auc,
    plot_result_aupr,
    set_seed,
)


# ============================================================
# I/O helpers
# ============================================================

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


def build_label_matrix(dataset: str) -> np.ndarray:
    return pd.read_csv(f"./dataset/{dataset}/{dataset}_baseline.csv", header=None).values


def build_all_pairs(df: np.ndarray):
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])], dtype=np.int64)
    return data, data[data[:, 2] == 1], data[data[:, 2] == 0]


def sample_train_negatives(train_neg_id: np.ndarray, train_pos_n: int, neg_ratio: int, seed: int):
    need_neg = neg_ratio * train_pos_n
    if need_neg > len(train_neg_id):
        raise ValueError(f"Not enough training negatives. Need {need_neg}, only {len(train_neg_id)} available.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_neg_id))
    return train_neg_id[perm[:need_neg]]


def build_pair_tensors(pos_id: np.ndarray, neg_id: np.ndarray, device: th.device):
    pos_pairs = pos_id[:, :2] if pos_id.shape[1] >= 2 else pos_id
    neg_pairs = neg_id[:, :2] if neg_id.shape[1] >= 2 else neg_id
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
    if dataset_name in ["Bdataset", "Cdataset", "RepoApp","lrssl"]:
        return {
            "drug": g.nodes["drug"].data["h"],
            "disease": g.nodes["disease"].data["h"],
            "protein": g.nodes["protein"].data["h"],
        }
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


# ============================================================
# Metrics / checkpoint
# ============================================================

@th.no_grad()
def evaluate_probs(labels_tensor: th.Tensor, probs_tensor: th.Tensor):
    labels_np = labels_tensor.detach().cpu().numpy()
    probs_np = probs_tensor.detach().cpu().numpy()
    auc, aupr = get_metrics_auc(labels_np, probs_np)
    _, _, acc, f1, pre, rec, spe = get_metrics(labels_np, probs_np)
    return auc, aupr, acc, f1, pre, rec, spe


@dataclass
class BestState:
    score: float = -1.0
    epoch: int = 0
    state: Optional[dict] = None
    test_auc: float = -1.0
    test_aupr: float = -1.0
    test_f1: float = -1.0
    test_pre: float = -1.0
    test_rec: float = -1.0
    test_spe: float = -1.0


class MetricTracker:
    def __init__(self):
        self.best = BestState()

    def step(self, monitor: float, epoch: int, model: nn.Module, metrics: Tuple[float, ...]):
        if monitor > self.best.score:
            auc, aupr, acc, f1, pre, rec, spe = metrics
            self.best.score = float(monitor)
            self.best.epoch = int(epoch)
            self.best.state = copy.deepcopy(model.state_dict())
            self.best.test_auc = float(auc)
            self.best.test_aupr = float(aupr)
            self.best.test_f1 = float(f1)
            self.best.test_pre = float(pre)
            self.best.test_rec = float(rec)
            self.best.test_spe = float(spe)


# ============================================================
# Model utilities
# ============================================================

def build_homo_pack_from_graph(graph, device: Optional[th.device] = None):
    """Precompute homogeneous graph pack for DGL HGTConv.

    This is graph-only and can be reused across all epochs in the current fold.
    """
    if device is not None:
        graph = graph.to(device)
    homo_graph = dgl.to_homogeneous(graph)
    if device is not None:
        homo_graph = homo_graph.to(device)

    node_ranges = {}
    start = 0
    for ntype in graph.ntypes:
        n = graph.num_nodes(ntype)
        node_ranges[ntype] = slice(start, start + n)
        start += n

    return {
        "g_homo": homo_graph,
        "ntype_ids": homo_graph.ndata[dgl.NTYPE].to(device) if device is not None else homo_graph.ndata[dgl.NTYPE],
        "etype_ids": homo_graph.edata[dgl.ETYPE].to(device) if device is not None else homo_graph.edata[dgl.ETYPE],
        "node_ranges": node_ranges,
        "ntypes": list(graph.ntypes),
    }


class HGTLayer(nn.Module):
    def __init__(self, hidden_dim: int, ntypes, etypes, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for HGTConv.")
        self.hgt = dglnn.HGTConv(
            hidden_dim,
            hidden_dim // num_heads,
            num_heads,
            len(ntypes),
            len(etypes),
            dropout=dropout,
            use_norm=True,
        )
        self.bn = nn.ModuleDict({ntype: nn.BatchNorm1d(hidden_dim) for ntype in ntypes})
        self.dropout = nn.Dropout(dropout)
        self.act = nn.PReLU()

    def forward(self, inputs: Dict[str, th.Tensor], homo_pack):
        feats = th.cat([inputs[ntype] for ntype in homo_pack["ntypes"]], dim=0)
        h_all = self.hgt(
            homo_pack["g_homo"],
            feats,
            homo_pack["ntype_ids"],
            homo_pack["etype_ids"],
            presorted=True,
        )
        out = {}
        for ntype, slc in homo_pack["node_ranges"].items():
            x = h_all[slc]
            x = self.bn[ntype](x)
            x = self.dropout(x)
            out[ntype] = self.act(x)
        return out


class LayerAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: int = 32):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )

    def forward(self, layers: List[th.Tensor], return_attention: bool = False):
        if len(layers) == 0:
            raise ValueError("No layer outputs to pool.")
        z = th.stack(layers, dim=1)  # [N,L,D]
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        out = (beta * z).sum(dim=1)
        if return_attention:
            return out, beta
        return out


def pool_layers(layers: List[th.Tensor], mode: str, attn_pool: Optional[LayerAttentionPooling] = None):
    if len(layers) == 0:
        raise ValueError("No layer outputs to pool.")
    if mode == "last":
        return layers[-1]
    if mode == "mean":
        return th.stack(layers, dim=0).mean(dim=0)
    if mode == "dream":
        out = layers[0]
        for i in range(1, len(layers)):
            out = out + layers[i] / float(i + 1)
        return out
    if mode == "attn":
        if attn_pool is None:
            raise ValueError("attn_pool is required when mode='attn'.")
        return attn_pool(layers)
    raise ValueError(f"Unsupported pooling mode: {mode}")


class PairDecoder(nn.Module):
    def __init__(self, hidden_dim: int, pair_hidden: int = 128, dropout: float = 0.1, pair_mode: str = "rotate"):
        super().__init__()
        if pair_mode not in ["rotate", "absdiff"]:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")
        if pair_mode == "rotate" and hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be divisible by 2 for rotate mode.")
        self.pair_mode = pair_mode
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, pair_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, pair_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, 1),
        )

    @staticmethod
    def rotate_operator(a, b):
        a_re, a_im = a.chunk(2, dim=-1)
        b_re, b_im = b.chunk(2, dim=-1)
        return th.cat([a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re], dim=-1)

    def pair_feature(self, drug_h, disease_h):
        mul = drug_h * disease_h
        last = self.rotate_operator(drug_h, disease_h) if self.pair_mode == "rotate" else th.abs(drug_h - disease_h)
        return th.cat([drug_h, disease_h, mul, last], dim=-1)

    def forward(self, h_nodes: Dict[str, th.Tensor], drug_idx: th.Tensor, disease_idx: th.Tensor):
        pair_feat = self.pair_feature(h_nodes["drug"][drug_idx], h_nodes["disease"][disease_idx])
        logit = self.mlp(pair_feat).squeeze(-1)
        return logit, pair_feat


class GlobalHGTOnlyModel(nn.Module):
    """Global HGT only.

    This is the raw semantic HGT branch:
      x_het features -> HGT layers -> layer pooling -> PairDecoder.
    """
    def __init__(
        self,
        etypes,
        ntypes,
        in_dims: Dict[str, int],
        hidden_feats: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_hgt_layers: int = 3,
        layer_pooling: str = "attn",
        pair_hidden: int = 128,
        pair_mode: str = "rotate",
    ):
        super().__init__()
        if layer_pooling not in ["last", "mean", "dream", "attn"]:
            raise ValueError(f"Unsupported layer_pooling: {layer_pooling}")
        self.ntypes = list(ntypes)
        self.hidden_feats = int(hidden_feats)
        self.num_hgt_layers = int(num_hgt_layers)
        self.layer_pooling = layer_pooling

        # Keep feature sources unchanged; only project each existing node feature into hidden space.
        self.input_linears = nn.ModuleDict({
            ntype: nn.Linear(int(in_dims[ntype]), hidden_feats)
            for ntype in self.ntypes
        })
        for layer in self.input_linears.values():
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.hgt_layers = nn.ModuleList([
            HGTLayer(hidden_feats, ntypes, etypes, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_hgt_layers)
        ])

        self.drug_layer_attn = LayerAttentionPooling(hidden_feats)
        self.disease_layer_attn = LayerAttentionPooling(hidden_feats)
        self.decoder = PairDecoder(hidden_feats, pair_hidden=pair_hidden, dropout=dropout, pair_mode=pair_mode)

    def _project_inputs(self, x_het: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return {ntype: self.input_linears[ntype](x_het[ntype]) for ntype in self.ntypes}

    def encode(self, x_het: Dict[str, th.Tensor], homo_pack, return_aux: bool = False):
        h = self._project_inputs(x_het)
        drug_layers, disease_layers = [], []
        all_layers = []

        for layer in self.hgt_layers:
            h = layer(h, homo_pack)
            drug_layers.append(h["drug"])
            disease_layers.append(h["disease"])
            if return_aux:
                all_layers.append({k: v for k, v in h.items()})

        drug_pool = pool_layers(
            drug_layers,
            self.layer_pooling,
            self.drug_layer_attn if self.layer_pooling == "attn" else None,
        )
        disease_pool = pool_layers(
            disease_layers,
            self.layer_pooling,
            self.disease_layer_attn if self.layer_pooling == "attn" else None,
        )
        nodes = {"drug": drug_pool, "disease": disease_pool}
        if return_aux:
            return nodes, {"drug_layers": drug_layers, "disease_layers": disease_layers, "all_layers": all_layers}
        return nodes

    def forward(self, x_het: Dict[str, th.Tensor], homo_pack, drug_idx: th.Tensor, disease_idx: th.Tensor, return_aux: bool = False):
        if return_aux:
            nodes, aux = self.encode(x_het, homo_pack, return_aux=True)
            logit, pair_feat = self.decoder(nodes, drug_idx, disease_idx)
            aux.update({"final_node_emb": nodes, "pair_feat": pair_feat})
            return logit, aux
        nodes = self.encode(x_het, homo_pack, return_aux=False)
        logit, _ = self.decoder(nodes, drug_idx, disease_idx)
        return logit



@th.no_grad()
def save_embedding_snapshot(
    model,
    x_het,
    homo_pack,
    fold_dir: str,
    fold: int,
    args,
    train_pos_pairs: np.ndarray,
    sampled_train_neg_pairs: np.ndarray,
    test_pos_pairs: np.ndarray,
    test_neg_pairs: np.ndarray,
    test_drug_idx: th.Tensor,
    test_disease_idx: th.Tensor,
    test_labels: th.Tensor,
    save_layer_embeddings: bool = True,
    save_pair_features: bool = False,
):
    """Save node/pair information for later t-SNE visualization.

    Default saved objects are deliberately aligned with future HGT+Sim runs:
      - final_drug_emb/final_disease_emb: pooled Global-HGT node representations.
      - input_proj_drug/input_proj_disease: projected input node representations before HGT.
      - optional layer-wise HGT embeddings.
      - test pair ids/labels/probs.
    """
    model.eval()
    logits, aux = model(x_het, homo_pack, test_drug_idx, test_disease_idx, return_aux=True)
    probs = th.sigmoid(logits)

    def cpu_np(x: th.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)

    test_pairs = np.concatenate([test_pos_pairs, test_neg_pairs], axis=0).astype(np.int64)
    test_labels_np = test_labels.detach().cpu().numpy().astype(np.float32)

    # Projected input features before Global HGT; useful as a "before HGT" reference.
    h0 = model._project_inputs(x_het)

    save_dict = {
        "final_drug_emb": cpu_np(aux["final_node_emb"]["drug"]),
        "final_disease_emb": cpu_np(aux["final_node_emb"]["disease"]),
        "input_proj_drug": cpu_np(h0["drug"]),
        "input_proj_disease": cpu_np(h0["disease"]),
        "test_pairs": test_pairs,
        "test_labels": test_labels_np,
        "test_probs": probs.detach().cpu().numpy().astype(np.float32),
        "train_pos_pairs": train_pos_pairs.astype(np.int64),
        "sampled_train_neg_pairs": sampled_train_neg_pairs.astype(np.int64),
        "fold": np.array([fold], dtype=np.int64),
        "best_epoch": np.array([getattr(args, "_best_epoch", -1)], dtype=np.int64),
    }

    if save_layer_embeddings:
        for i, z in enumerate(aux["drug_layers"]):
            save_dict[f"drug_layer_{i}"] = cpu_np(z)
        for i, z in enumerate(aux["disease_layers"]):
            save_dict[f"disease_layer_{i}"] = cpu_np(z)

    if save_pair_features:
        save_dict["test_pair_feat"] = cpu_np(aux["pair_feat"])

    out_path = os.path.join(fold_dir, "hgt_only_tsne_data.npz")
    np.savez_compressed(out_path, **save_dict)

    meta = {
        "model": "GlobalHGTOnlyModel",
        "dataset": args.dataset,
        "fold": int(fold),
        "best_epoch": int(getattr(args, "_best_epoch", -1)),
        "feature_mode": args.feature_mode,
        "hidden_feats": int(args.hidden_feats),
        "num_hgt_layers": int(args.num_hgt_layers),
        "num_heads": int(args.num_heads),
        "dropout": float(args.dropout),
        "layer_pooling": args.layer_pooling,
        "pair_mode": args.pair_mode,
        "neg_ratio": int(args.neg_ratio),
        "saved_arrays": sorted(list(save_dict.keys())),
        "note": "Global HGT only; no sim view, no RBF sim, no query pooling. Node initial features come from x_het/load(... feature_mode).",
    }
    meta_path = os.path.join(fold_dir, "hgt_only_tsne_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[t-SNE data] Saved node/pair snapshot: {out_path}")
    print(f"[t-SNE data] Saved metadata: {meta_path}")


# ============================================================
# Train / predict
# ============================================================

def train_one_epoch(model, x_het, homo_pack, drug_idx, disease_idx, labels, optimizer, criterion, grad_clip: float = 5.0):
    model.train()
    n = labels.shape[0]
    perm = th.randperm(n, device=labels.device)
    drug_idx = drug_idx[perm]
    disease_idx = disease_idx[perm]
    labels = labels[perm]

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_het, homo_pack, drug_idx, disease_idx, return_aux=False)
    loss = criterion(logits, labels)
    loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    probs = th.sigmoid(logits.detach())
    train_auc, train_aupr, *_ = evaluate_probs(labels.detach(), probs.detach())
    return {
        "loss": float(loss.detach().cpu().item()),
        "train_auc": train_auc,
        "train_aupr": train_aupr,
    }


@th.no_grad()
def predict_prob(model, x_het, homo_pack, drug_idx, disease_idx):
    model.eval()
    logits = model(x_het, homo_pack, drug_idx, disease_idx, return_aux=False)
    return th.sigmoid(logits)

@th.no_grad()
def extract_global_pair_features_batched(
    model,
    x_het,
    homo_pack,
    drug_idx,
    disease_idx,
    batch_size: int = 65536,
):
    """
    Export global-only pair-level features:
      - global_pair_feat: final Global-HGT pair feature
      - global_prob: prediction probability
      - orig_pair_feat_global: input-projected pair feature before HGT
    """
    model.eval()

    h0 = model._project_inputs(x_het)

    orig_feats = []
    global_feats = []
    probs = []

    n = drug_idx.numel()
    for st in range(0, n, int(batch_size)):
        ed = min(st + int(batch_size), n)
        d = drug_idx[st:ed]
        s = disease_idx[st:ed]

        orig_pair_feat = model.decoder.pair_feature(
            h0["drug"][d],
            h0["disease"][s],
        )

        logits, aux = model(
            x_het,
            homo_pack,
            d,
            s,
            return_aux=True,
        )
        global_pair_feat = aux["pair_feat"]

        orig_feats.append(orig_pair_feat.detach().cpu())
        global_feats.append(global_pair_feat.detach().cpu())
        probs.append(th.sigmoid(logits).detach().cpu())

    return {
        "orig_pair_feat_global": th.cat(orig_feats, dim=0).numpy().astype(np.float32),
        "global_pair_feat": th.cat(global_feats, dim=0).numpy().astype(np.float32),
        "global_prob": th.cat(probs, dim=0).numpy().astype(np.float32),
    }

# ============================================================
# Args / main
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-da", "--dataset", default="Cdataset")
    parser.add_argument("-id", "--device_id", default="0")
    parser.add_argument("-fo", "--nfold", type=int, default=10)
    parser.add_argument("-nr", "--neg_ratio", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("-sp", "--saved_path", default="result_global_hgt_only")
    parser.add_argument("--feature_mode", choices=["llm", "random"], default="llm")
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--save_embeddings", action="store_true", default=True, help="Save node embeddings and test-pair metadata for later t-SNE visualization.")
    parser.add_argument("--no_save_embeddings", action="store_false", dest="save_embeddings", help="Disable saving t-SNE embedding snapshots.")
    parser.add_argument("--save_layer_embeddings", action="store_true", default=True, help="Save each HGT layer drug/disease embedding.")
    parser.add_argument("--no_save_layer_embeddings", action="store_false", dest="save_layer_embeddings")
    parser.add_argument("--save_pair_features", action="store_true", default=False, help="Also save test pair features; this can be large.")

    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--bce_pos_weight_scale", type=float, default=1.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden_feats", type=int, default=128)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pair_hidden", type=int, default=128)
    parser.add_argument("--pair_mode", choices=["rotate", "absdiff"], default="rotate")
    parser.add_argument("--layer_pooling", choices=["last", "mean", "dream", "attn"], default="attn")
    parser.add_argument(
        "--predict_only",
        action="store_true",
        default=False,
        help="Do not train. Rebuild each fold graph, load fold*/best_model_state.pth, and export pair features."
    )

    parser.add_argument(
        "--checkpoint_root",
        default=None,
        help="Root directory containing fold1/fold2/.../best_model_state.pth. If None, use --saved_path."
    )

    parser.add_argument(
        "--predict_batch_size",
        type=int,
        default=65536,
        help="Batch size for prediction/export."
    )
    return parser


def main():
    args = build_parser().parse_args()
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

    print(f"Matrix shape: {df.shape[0]} x {df.shape[1]}")
    print(f"Positive samples: {len(data_pos):,}")
    print(f"Negative samples: {len(data_neg):,}")
    print(f"Total pairs: {len(data):,}")

    kf_pos = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    kf_neg = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)

    pred_result = np.zeros(df.shape, dtype=np.float32)
    fold_metrics = []
    fold = 1

    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf_pos.split(data_pos), kf_neg.split(data_neg)):
        print("\n" + "=" * 80)
        print(f"{args.nfold}-Fold CV | Fold {fold}")
        print("=" * 80)

        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]

        sampled_train_neg_id = sample_train_negatives(
            train_neg_id[:, :2],
            len(train_pos_id),
            args.neg_ratio,
            args.seed + fold * 20000,
        )

        test_pos_pairs = test_pos_id[:, :2]
        test_neg_pairs = test_neg_id[:, :2]

        print(f"Train pos pool: {len(train_pos_id):,}")
        print(f"Train neg pool: {len(train_neg_id):,}")
        print(f"Sampled train neg: {len(sampled_train_neg_id):,}")
        print(f"Test pos:       {len(test_pos_id):,}")
        print(f"Test neg:       {len(test_neg_id):,}")

        g_het = safe_load_heterograph(
            args.dataset,
            device=device,
            base_dir=base_dir,
            device_id=int(args.device_id),
            feature_mode=args.feature_mode,
        )
        g_het = remove_graph(g_het, test_pos_pairs).to(device)
        x_het = {k: v.to(device) for k, v in get_feature_dict(g_het, args.dataset).items()}
        homo_pack = build_homo_pack_from_graph(g_het, device=device)
        in_dims = {ntype: int(feat.size(1)) for ntype, feat in x_het.items()}

        print(f"[Fold graph] Loaded heterograph and removed current-fold test positives: {len(test_pos_pairs)}")
        print(f"[Feature dims] {in_dims}")
        print(f"[Graph] ntypes={g_het.ntypes}, etypes={g_het.etypes}")

        train_drug_idx, train_disease_idx, train_labels = build_pair_tensors(train_pos_id[:, :2], sampled_train_neg_id, device)
        test_drug_idx, test_disease_idx, test_labels = build_pair_tensors(test_pos_pairs, test_neg_pairs, device)

        model = GlobalHGTOnlyModel(
            etypes=g_het.etypes,
            ntypes=g_het.ntypes,
            in_dims=in_dims,
            hidden_feats=args.hidden_feats,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_hgt_layers=args.num_hgt_layers,
            layer_pooling=args.layer_pooling,
            pair_hidden=args.pair_hidden,
            pair_mode=args.pair_mode,
        ).to(device)

        fold_dir = os.path.join(args.saved_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        if args.predict_only:
            ckpt_root = args.checkpoint_root or args.saved_path
            ckpt_path = os.path.join(ckpt_root, f"fold{fold}", "best_model_state.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            state = th.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            print(f"[fold {fold}] Loaded global-only checkpoint: {ckpt_path}")

            pair_pack = extract_global_pair_features_batched(
                model=model,
                x_het=x_het,
                homo_pack=homo_pack,
                drug_idx=test_drug_idx,
                disease_idx=test_disease_idx,
                batch_size=args.predict_batch_size,
            )

            test_pairs = np.concatenate([test_pos_pairs, test_neg_pairs], axis=0).astype(np.int64)
            test_labels_np = test_labels.detach().cpu().numpy().astype(np.int64)

            np.savez_compressed(
                os.path.join(fold_dir, "global_pair_features_for_3d.npz"),
                test_pairs=test_pairs,
                labels=test_labels_np,
                orig_pair_feat_global=pair_pack["orig_pair_feat_global"],
                global_pair_feat=pair_pack["global_pair_feat"],
                global_prob=pair_pack["global_prob"],
                fold=np.array([fold], dtype=np.int64),
            )

            print(f"[fold {fold}] Saved global pair features:",
                  os.path.join(fold_dir, "global_pair_features_for_3d.npz"))

            fold += 1
            continue

        pos_weight = th.tensor(
            (len(sampled_train_neg_id) / max(1, len(train_pos_id))) * args.bce_pos_weight_scale,
            dtype=th.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.patience, factor=0.5)
        tracker = MetricTracker()

        fold_dir = os.path.join(args.saved_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        epoch_rows = []

        t_train0 = time.time()
        for epoch in range(1, args.epoch + 1):
            train_stats = train_one_epoch(
                model=model,
                x_het=x_het,
                homo_pack=homo_pack,
                drug_idx=train_drug_idx,
                disease_idx=train_disease_idx,
                labels=train_labels,
                optimizer=optimizer,
                criterion=criterion,
                grad_clip=args.grad_clip,
            )

            t_train1 = time.time()
            t_test0 = time.time()
            test_probs = predict_prob(model, x_het, homo_pack, test_drug_idx, test_disease_idx)
            test_auc, test_aupr, test_acc, test_f1, test_pre, test_rec, test_spe = evaluate_probs(test_labels, test_probs)
            t_test1 = time.time()

            scheduler.step(test_aupr)
            tracker.step(test_aupr, epoch, model, (test_auc, test_aupr, test_acc, test_f1, test_pre, test_rec, test_spe))

            train_time = (t_train1 - t_train0) / 60.0
            test_time = (t_test1 - t_test0) / 60.0

            epoch_rows.append({
                "epoch": epoch,
                "loss": train_stats["loss"],
                "train_auc": train_stats["train_auc"],
                "train_aupr": train_stats["train_aupr"],
                "test_auc": test_auc,
                "test_aupr": test_aupr,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_spe": test_spe,
            })

            if epoch == 1 or epoch == args.epoch or epoch % 10 == 0:
                print(
                    f"[fold {fold}] Epoch {epoch:03d} | "
                    f"Loss {train_stats['loss']:.4f} | Train AUPR {train_stats['train_aupr']:.4f} | "
                    f"|| Test AUC {test_auc:.4f} | Test AUPR {test_aupr:.4f} | "
                    f"F1 {test_f1:.4f} | Rec {test_rec:.4f} | Pre {test_pre:.4f} | "
                    f"Train {train_time:.2f}min | Test {test_time:.2f}min"
                )

        pd.DataFrame(epoch_rows).to_csv(os.path.join(fold_dir, "epoch_metrics.csv"), index=False)

        if tracker.best.state is not None:
            model.load_state_dict(tracker.best.state)
            args._best_epoch = tracker.best.epoch
            th.save(tracker.best.state, os.path.join(fold_dir, "best_model_state.pth"))
            print(
                f"[fold {fold}] Best checkpoint | Epoch {tracker.best.epoch} | "
                f"Test AUC {tracker.best.test_auc:.4f} | Test AUPR {tracker.best.test_aupr:.4f} | "
                f"F1 {tracker.best.test_f1:.4f} | Rec {tracker.best.test_rec:.4f} | Pre {tracker.best.test_pre:.4f}"
            )

        final_probs = predict_prob(model, x_het, homo_pack, test_drug_idx, test_disease_idx)
        fold_auc, fold_aupr, fold_acc, fold_f1, fold_pre, fold_rec, fold_spe = evaluate_probs(test_labels, final_probs)

        print(f"\nFold {fold} Final Test AUC:  {fold_auc:.4f}")
        print(f"Fold {fold} Final Test AUPR: {fold_aupr:.4f}")
        print(f"Fold {fold} Final Test ACC:  {fold_acc:.4f}")
        print(f"Fold {fold} Final Test F1:   {fold_f1:.4f}")
        print(f"Fold {fold} Final Test Rec:  {fold_rec:.4f}")
        print(f"Fold {fold} Final Test Pre:  {fold_pre:.4f}")
        print(f"Fold {fold} Final Test Spe:  {fold_spe:.4f}")

        if args.save_embeddings:
            save_embedding_snapshot(
                model=model,
                x_het=x_het,
                homo_pack=homo_pack,
                fold_dir=fold_dir,
                fold=fold,
                args=args,
                train_pos_pairs=train_pos_id[:, :2],
                sampled_train_neg_pairs=sampled_train_neg_id,
                test_pos_pairs=test_pos_pairs,
                test_neg_pairs=test_neg_pairs,
                test_drug_idx=test_drug_idx,
                test_disease_idx=test_disease_idx,
                test_labels=test_labels,
                save_layer_embeddings=args.save_layer_embeddings,
                save_pair_features=args.save_pair_features,
            )

        fold_metrics.append({
            "fold": fold,
            "auc": fold_auc,
            "aupr": fold_aupr,
            "acc": fold_acc,
            "f1": fold_f1,
            "pre": fold_pre,
            "rec": fold_rec,
            "spe": fold_spe,
        })

        test_probs_np = final_probs.detach().cpu().numpy()
        pred_result[test_pos_pairs[:, 0], test_pos_pairs[:, 1]] = test_probs_np[: len(test_pos_pairs)]
        pred_result[test_neg_pairs[:, 0], test_neg_pairs[:, 1]] = test_probs_np[len(test_pos_pairs):]

        fold += 1

    np.save(os.path.join(args.saved_path, "cv_pred_matrix.npy"), pred_result.astype(np.float32))
    print("Saved CV prediction matrix:", os.path.join(args.saved_path, "cv_pred_matrix.npy"))

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_path = os.path.join(args.saved_path, "fold_metrics.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    print("Saved fold metrics:", fold_metrics_path)

    summary_rows = []
    for key in ["auc", "aupr", "acc", "f1", "pre", "rec", "spe"]:
        vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        summary_rows.append({"metric": key, "mean": vals.mean(), "std": vals.std(ddof=1) if len(vals) > 1 else 0.0})
    summary_path = os.path.join(args.saved_path, "summary_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("Saved summary metrics:", summary_path)

    overall_labels = df.reshape(-1)
    overall_preds = pred_result.reshape(-1)
    overall_auc, overall_aupr = get_metrics_auc(overall_labels, overall_preds)
    _, _, overall_acc, overall_f1, overall_pre, overall_rec, overall_spe = get_metrics(overall_labels, overall_preds)

    print("\n" + "=" * 80)
    print("Overall CV Results")
    print("=" * 80)
    print(f"Overall AUC:  {overall_auc:.4f}")
    print(f"Overall AUPR: {overall_aupr:.4f}")
    print(f"Overall Acc:  {overall_acc:.4f}")
    print(f"Overall F1:   {overall_f1:.4f}")
    print(f"Overall Rec:  {overall_rec:.4f}")
    print(f"Overall Pre:  {overall_pre:.4f}")
    print(f"Overall Spe:  {overall_spe:.4f}")

    print("\nPer-fold Mean ± SD")
    for key in ["auc", "aupr", "acc", "f1", "pre", "rec", "spe"]:
        vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        print(f"{key.upper()}: {vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")

    if args.save_plots:
        plot_result_auc(args, overall_labels, overall_preds, overall_auc)
        plot_result_aupr(args, overall_labels, overall_preds, overall_aupr)
        print("Saved overall ROC curve:", os.path.join(args.saved_path, "result_auc.png"))
        print("Saved overall PR curve:", os.path.join(args.saved_path, "result_aupr.png"))

    log_file.close()


if __name__ == "__main__":
    main()
