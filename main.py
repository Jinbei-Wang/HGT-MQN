import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import KFold

from load_data2 import load, remove_graph, prepare_similarity_graphs
from utiles.utils import (
    get_metrics_auc,
    get_metrics,
    plot_result_auc,
    plot_result_aupr,
    set_seed,
)
# from model_update3_siminit_fixed import HGTSimGTQueryCLModel
from model_update5 import HGTSimGTQueryCLModel
# from model_update5_2 import HGTSimGTQueryCLModel#rdb-sim


# ============================================================
# Data helpers
# ============================================================

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


# ============================================================
# Loss / metrics / checkpoint tracker
# ============================================================

def build_binary_criterion(pos_weight: th.Tensor):
    return th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)#这里我稍微更改下


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
# Train / prediction
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


@th.no_grad()
def predict_prob(model, g_het, x_het, dr_graph, di_graph, drug_idx, disease_idx):
    model.eval()
    logits = model_forward(model, g_het, x_het, dr_graph, di_graph, drug_idx, disease_idx, return_aux=False)
    if isinstance(logits, tuple):
        logits = logits[0]
    return th.sigmoid(logits)


@th.no_grad()
def predict_hgt_raw_prob(model, g_het, x_het, drug_idx, disease_idx):
    model.eval()
    logits = model.forward_hgt_raw(g_het, x_het, drug_idx, disease_idx)
    return th.sigmoid(logits)

def build_grouped_optimizer(model, args):
    """
    Separate learning rates for:
      1) HGT/global branch
      2) Similarity GT branch
      3) Query/pooling/decoder branch
    """

    base_lr = args.learning_rate
    hgt_lr = args.hgt_learning_rate if args.hgt_learning_rate is not None else base_lr
    sim_lr = args.sim_learning_rate if args.sim_learning_rate is not None else base_lr
    query_lr = args.query_learning_rate if args.query_learning_rate is not None else base_lr

    # -------- HGT / global branch --------
    hgt_params = []
    hgt_params += list(model.drug_linear.parameters())
    hgt_params += list(model.disease_linear.parameters())
    hgt_params += list(model.other_linear.parameters())
    hgt_params += list(model.hgt_layers.parameters())

    # -------- Sim-GT branch --------
    sim_params = []
    sim_params += list(model.sim_encoder.parameters())
    if hasattr(model, "sim_drug_feature_linear"):
        sim_params += list(model.sim_drug_feature_linear.parameters())
    if hasattr(model, "sim_disease_feature_linear"):
        sim_params += list(model.sim_disease_feature_linear.parameters())

    # -------- Query / pooling / decoder branch --------
    query_params = []
    query_params += list(model.query_blocks.parameters())
    query_params += list(model.drug_layer_attn.parameters())
    query_params += list(model.disease_layer_attn.parameters())
    query_params += list(model.sim_drug_layer_attn.parameters())
    query_params += list(model.sim_disease_layer_attn.parameters())
    query_params += list(model.pair_decoder.parameters())

    optimizer = th.optim.Adam(
        [
            {
                "params": hgt_params,
                "lr": hgt_lr,
                "weight_decay": args.weight_decay,
                "name": "hgt_global",
            },
            {
                "params": sim_params,
                "lr": sim_lr,
                "weight_decay": args.weight_decay,
                "name": "sim_gt",
            },
            {
                "params": query_params,
                "lr": query_lr,
                "weight_decay": args.weight_decay,
                "name": "query_decoder",
            },
        ]
    )

    print(
        f"[Optimizer] lr setting | "
        f"HGT/global={hgt_lr:g}, "
        f"Sim-GT={sim_lr:g}, "
        f"Query/decoder={query_lr:g}, "
        f"weight_decay={args.weight_decay:g}"
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
    lambda_cl_het: float = 0.001,
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
        model,
        g_het,
        x_het,
        dr_graph,
        di_graph,
        drug_idx,
        disease_idx,
        return_aux=True,
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

    query_gamma = aux.get("query_gamma_mean", None)
    drug_gate = aux.get("drug_query_gate_mean", None)
    disease_gate = aux.get("disease_query_gate_mean", None)

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
        "query_gamma": to_float(query_gamma),
        "drug_gate": to_float(drug_gate),
        "disease_gate": to_float(disease_gate),
        "train_auc": train_auc,
        "train_aupr": train_aupr,
    }


# ============================================================
# Args
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-da", "--dataset", default="Cdataset")
    parser.add_argument("-id", "--device_id", default="0")
    parser.add_argument("-fo", "--nfold", type=int, default=10)
    parser.add_argument("-nr", "--neg_ratio", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("-sp", "--saved_path", default="result_hgt_simgcn_query_cl")
    parser.add_argument("--feature_mode", choices=["llm", "random"], default="llm")
    parser.add_argument(
        "--sim_init_mode",
        choices=["sim_feature", "shared"],
        default="sim_feature",
        help=(
            "SimGT initial feature source. 'sim_feature' uses dr_graph/di_graph.ndata['sim_feature'] "
            "from baseline similarity profiles; 'shared' reuses HGT projected drug/disease features for ablation."
        ),
    )
    parser.add_argument("--save_plots", action="store_true", default=True)

    # training
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--bce_pos_weight_scale", type=float, default=1.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # model
    parser.add_argument("--hidden_feats", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    # Set both HGT/global and sim GCN view layers to 2 by default.
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_sim_layers", type=int, default=2)
    parser.add_argument(
        "--query_layers",
        type=str,
        default="1",
        help="HGT layers using sim query pooling. Use 'all', 'none', or comma-separated 0-based ids, e.g. '0,1'.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pair_hidden", type=int, default=128)
    parser.add_argument("--pair_mode", choices=["rotate", "absdiff"], default="rotate")
    parser.add_argument("--layer_pooling", choices=["last", "mean", "dream", "attn"], default="attn")

    parser.add_argument(
        "--hgt_learning_rate",
        type=float,
        default=None,
        help="Learning rate for HGT/global branch. If None, use --learning_rate."
    )

    parser.add_argument(
        "--sim_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for similarity GT branch. If None, use --learning_rate."
    )

    parser.add_argument(
        "--query_learning_rate",
        type=float,
        default=None,
        help="Learning rate for query blocks, pooling layers and decoder. If None, use --learning_rate."
    )

    # similarity GCN / query pooling
    parser.add_argument("--sim_topk", type=int, default=5)
    parser.add_argument("--sim_use_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_no_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_diffusion_alpha", type=float, default=0.15)
    parser.add_argument("--sim_diffusion_steps", type=int, default=3)
    parser.add_argument("--sim_use_diffused_adj_for_gcn", action="store_true", default=False)
    parser.add_argument("--query_gamma_init", type=float, default=0.05)

    # contrastive learning: separated lambdas
    parser.add_argument("--lambda_cl_het", type=float, default=0.1,
                        help="InfoNCE weight: query-enhanced/fused global representation vs raw HGT representation.")
    parser.add_argument("--lambda_cl_sim", type=float, default=0.0,
                        help="InfoNCE weight: query-enhanced/fused global representation vs sim-GCN representation.")
    parser.add_argument("--cl_temperature", type=float, default=0.2)
    parser.add_argument("--cl_sample_size", type=int, default=0,
                        help="If >0, sample this many nodes for each InfoNCE matrix to reduce memory.")

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

    print(f"Matrix shape: {df.shape[0]} x {df.shape[1]}")
    print(f"Positive samples: {len(data_pos):,}")
    print(f"Negative samples: {len(data_neg):,}")
    print(f"Total pairs: {len(data):,}")

    dr_graph, di_graph = prepare_similarity_graphs(
        dataset=args.dataset,
        base_dir=base_dir,
        K=args.sim_topk,
        device=device,
        make_undirected=True,
    )
    n_drug, n_dis = dr_graph.num_nodes(), di_graph.num_nodes()
    print(f"[Similarity Graphs] n_drug={n_drug}, n_dis={n_dis}, sim_topk={args.sim_topk}")

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

        print(f"Train pos pool: {len(train_pos_id):,}")
        print(f"Train neg pool: {len(train_neg_id):,}")
        print(f"Test pos:       {len(test_pos_id):,}")
        print(f"Test neg:       {len(test_neg_id):,}")

        sampled_train_neg_id = sample_train_negatives(
            train_neg_id[:, :2],
            len(train_pos_id),
            args.neg_ratio,
            args.seed + fold * 20000,
        )
        print(f"Sampled train neg: {len(sampled_train_neg_id):,}")

        test_pos_pairs = test_pos_id[:, :2]
        test_neg_pairs = test_neg_id[:, :2]

        g_het = safe_load_heterograph(
            args.dataset,
            device=device,
            base_dir=base_dir,
            device_id=int(args.device_id),
            feature_mode=args.feature_mode,
        )
        g_het = remove_graph(g_het, test_pos_pairs).to(device)
        x_het = {k: v.to(device) for k, v in get_feature_dict(g_het, args.dataset).items()}
        print(f"[Fold graph] Loaded base heterograph and removed current fold test positives: {len(test_pos_pairs)}")

        model = HGTSimGTQueryCLModel(
            etypes=g_het.etypes,
            ntypes=g_het.ntypes,
            n_drug=n_drug,
            n_dis=n_dis,
            in_feats=args.hidden_feats,
            hidden_feats=args.hidden_feats,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_hgt_layers=args.num_hgt_layers,
            num_sim_layers=args.num_sim_layers,
            query_layers=args.query_layers,
            pair_hidden=args.pair_hidden,
            pair_mode=args.pair_mode,
            sim_init_mode=args.sim_init_mode,
            sim_drug_in_dim=n_drug,
            sim_disease_in_dim=n_dis,
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
        model.set_similarity_graphs(dr_graph, di_graph, device=device)
        print(f"[Sim-GT] Similarity graphs have been set. sim_init_mode={args.sim_init_mode}. No A_train is used in the sim branch.")

        train_drug_idx, train_disease_idx, train_labels = build_pair_tensors(train_pos_id[:, :2], sampled_train_neg_id, device)
        test_drug_idx, test_disease_idx, test_labels = build_pair_tensors(test_pos_pairs, test_neg_pairs, device)

        pos_weight = th.tensor(
            (len(sampled_train_neg_id) / max(1, len(train_pos_id))) * args.bce_pos_weight_scale,
            dtype=th.float32,
            device=device,
        )
        criterion = build_binary_criterion(pos_weight)
        # optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer = build_grouped_optimizer(model, args)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.patience, factor=0.5)
        tracker = MetricTracker()

        fold_dir = os.path.join(args.saved_path, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        t_train0 = time.time()
        for epoch in range(1, args.epoch + 1):
            train_stats = train_one_epoch(
                model=model,
                g_het=g_het,
                x_het=x_het,
                dr_graph=dr_graph,
                di_graph=di_graph,
                drug_idx=train_drug_idx,
                disease_idx=train_disease_idx,
                labels=train_labels,
                optimizer=optimizer,
                criterion=criterion,
                lambda_cl_het=args.lambda_cl_het,
                lambda_cl_sim=args.lambda_cl_sim,
                grad_clip=args.grad_clip,
            )

            t_train1 = time.time()
            t_test0 = time.time()
            test_probs = predict_prob(model, g_het, x_het, dr_graph, di_graph, test_drug_idx, test_disease_idx)
            test_auc, test_aupr, _, test_f1, test_pre, test_rec, test_spe = evaluate_probs(test_labels, test_probs)

            hgt_raw_probs = predict_hgt_raw_prob(model, g_het, x_het, test_drug_idx, test_disease_idx)
            _, hgt_raw_aupr, *_ = evaluate_probs(test_labels, hgt_raw_probs)

            t_test1 = time.time()
            train_time = (t_train1 - t_train0) / 60.0
            test_time = (t_test1 - t_test0) / 60.0

            monitor_score = test_aupr
            scheduler.step(test_aupr)
            tracker.step(monitor_score, epoch, model.state_dict(), test_auc, test_aupr, test_rec)

            if epoch == 1 or epoch == args.epoch or epoch % 10 == 0:
                print(
                    f"[fold {fold}] Epoch {epoch:03d} | "
                    f"Loss {train_stats['loss']:.4f} | Pred {train_stats['pred_loss']:.4f} | "
                    f"CLhet {train_stats['cl_het']:.4f}*{args.lambda_cl_het:g} | "
                    f"CLsim {train_stats['cl_sim']:.4f}*{args.lambda_cl_sim:g} | "
                    f"QGamma {train_stats['query_gamma']:.4f} | "
                    f"GateD/S {train_stats['drug_gate']:.3f}/{train_stats['disease_gate']:.3f} | "
                    f"Train AUPR {train_stats['train_aupr']:.4f} | "
                    f"|| Test AUC {test_auc:.4f} | Test AUPR {test_aupr:.4f} | HGT-raw AUPR {hgt_raw_aupr:.4f} | "
                    f"Test F1 {test_f1:.4f} | Test Recall {test_rec:.4f} | Test Pre {test_pre:.4f} | "
                    f"Train Time {train_time:.2f}min | Test Time {test_time:.2f}min"
                )

        if tracker.best.state is not None:
            model.load_state_dict(tracker.best.state)
            th.save(tracker.best.state, os.path.join(fold_dir, "best_model_state.pth"))
            print(
                f"[fold {fold}] Best checkpoint | Epoch {tracker.best.epoch} | "
                f"Test AUC {tracker.best.test_auc:.4f} | Test AUPR {tracker.best.test_aupr:.4f} | Rec {tracker.best.test_rec:.4f}"
            )

        final_probs = predict_prob(model, g_het, x_het, dr_graph, di_graph, test_drug_idx, test_disease_idx)
        fold_auc, fold_aupr, fold_acc, fold_f1, fold_pre, fold_rec, fold_spe = evaluate_probs(test_labels, final_probs)
        hgt_raw_probs = predict_hgt_raw_prob(model, g_het, x_het, test_drug_idx, test_disease_idx)
        _, hgt_raw_aupr, *_ = evaluate_probs(test_labels, hgt_raw_probs)

        print(f"\nFold {fold} Final Test AUC:      {fold_auc:.4f}")
        print(f"Fold {fold} Final Test AUPR:     {fold_aupr:.4f}")
        print(f"Fold {fold} HGT-raw Test AUPR:   {hgt_raw_aupr:.4f}")
        print(f"Fold {fold} Final Test ACC:      {fold_acc:.4f}")
        print(f"Fold {fold} Final Test F1:       {fold_f1:.4f}")
        print(f"Fold {fold} Final Test Rec:      {fold_rec:.4f}")
        print(f"Fold {fold} Final Test Pre:      {fold_pre:.4f}")
        print(f"Fold {fold} Final Test Spe:      {fold_spe:.4f}")

        fold_metrics.append({
            "fold": fold,
            "auc": fold_auc,
            "aupr": fold_aupr,
            "hgt_raw_aupr": hgt_raw_aupr,
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

    if len(fold_metrics) > 0:
        print("\nPer-fold Mean ± SD")
        for key in ["auc", "aupr", "hgt_raw_aupr", "acc", "f1", "pre", "rec", "spe"]:
            vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
            print(f"{key.upper()}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

    if args.save_plots:
        plot_result_auc(args, overall_labels, overall_preds, overall_auc)
        plot_result_aupr(args, overall_labels, overall_preds, overall_aupr)
        print("Saved overall ROC curve:", os.path.join(args.saved_path, "result_auc.png"))
        print("Saved overall PR curve:", os.path.join(args.saved_path, "result_aupr.png"))

    log_file.close()


if __name__ == "__main__":
    main()
