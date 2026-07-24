#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate RepoApp-trained strict cold-start models on RepoClin by external double-cold
pseudo embedding transfer.

This script removes the drug_cold / disease_cold mode switch for external testing.
RepoClin drugs and RepoClin diseases are both treated as unseen external entities:

  RepoClin drug    -> top-k similar RepoApp seen drugs    -> pseudo drug embedding
  RepoClin disease -> top-k similar RepoApp seen diseases -> pseudo disease embedding
  pseudo pair embeddings -> RepoApp-trained pair_decoder -> RepoClin score

Expected RepoApp checkpoint structure:
  <repoapp_ckpt_root>/fold1/best_model_state.pth
  ...
  <repoapp_ckpt_root>/fold10/best_model_state.pth

For external RepoClin testing, all RepoApp drugs and diseases are treated as
seen source entities. No strict_cold_metadata.json is required.

RepoClin association csv format:
  Drug,Disease
  5,0
  836,0
  ...

Labels:
  association csv pairs are positives; unlisted RepoClin drug-disease pairs are negatives.

Notes:
  - This script uses raw feature cosine retrieval only, because RepoClin is an external
    dataset and usually has no cross-dataset similarity matrix to RepoApp.
  - The model architecture arguments must match the training script that produced
    best_model_state.pth.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import dgl

from load_data2 import load, prepare_similarity_graphs
from model_update5 import HGTSimGTQueryCLModel
from utiles.utils import get_metrics_auc, get_metrics, set_seed


# ============================================================
# IO / logging
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


def safe_load_heterograph(dataset: str, device: th.device, base_dir=None, device_id=None, feature_mode="llm"):
    try:
        return load(dataset, base_dir=base_dir, device=device, device_id=device_id, feature_mode=feature_mode)
    except TypeError:
        return load(dataset)


def get_feature_dict(g) -> Dict[str, th.Tensor]:
    """Generic feature dict: use every node type that has ndata['h']."""
    out = {}
    for ntype in g.ntypes:
        if "h" in g.nodes[ntype].data:
            out[ntype] = g.nodes[ntype].data["h"]
    if "drug" not in out or "disease" not in out:
        raise KeyError("Graph must contain drug/disease node features in ndata['h'].")
    return out


def find_npy(path_or_dir: str, candidates: List[str], explicit_path: str = None) -> str:
    if explicit_path is not None and explicit_path != "":
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(explicit_path)
        return explicit_path
    for name in candidates:
        p = os.path.join(path_or_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find npy in {path_or_dir}. Tried: {candidates}. "
        f"Please pass explicit --repoclin_drug_npy / --repoclin_disease_npy."
    )


def load_repoclin_data(args):
    drug_npy = find_npy(
        args.repoclin_dir,
        [
            "drug_LLM_emb_aligned.npy", "drug_LLM_emb.npy", "drug_emb.npy",
            "repoclin_drug_LLM_emb_aligned.npy", "RepoClin_drug_LLM_emb_aligned.npy",
        ],
        args.repoclin_drug_npy,
    )
    disease_npy = find_npy(
        args.repoclin_dir,
        [
            "disease_LLM_emb_aligned.npy", "disease_LLM_emb.npy", "disease_emb.npy",
            "repoclin_disease_LLM_emb_aligned.npy", "RepoClin_disease_LLM_emb_aligned.npy",
        ],
        args.repoclin_disease_npy,
    )
    assoc_csv = args.repoclin_assoc_csv
    if not os.path.isfile(assoc_csv):
        raise FileNotFoundError(assoc_csv)

    drug_x = np.load(drug_npy).astype(np.float32)
    disease_x = np.load(disease_npy).astype(np.float32)

    assoc = pd.read_csv(assoc_csv)
    if "Drug" not in assoc.columns or "Disease" not in assoc.columns:
        # allow headerless two-column file
        if assoc.shape[1] >= 2:
            assoc = assoc.iloc[:, :2]
            assoc.columns = ["Drug", "Disease"]
        else:
            raise ValueError("RepoClin association csv must have Drug and Disease columns.")

    assoc["Drug"] = assoc["Drug"].astype(int)
    assoc["Disease"] = assoc["Disease"].astype(int)

    n_drug, n_dis = drug_x.shape[0], disease_x.shape[0]
    if assoc["Drug"].max() >= n_drug or assoc["Disease"].max() >= n_dis:
        raise ValueError(
            f"Association index out of range. max Drug={assoc['Drug'].max()}, n_drug={n_drug}; "
            f"max Disease={assoc['Disease'].max()}, n_disease={n_dis}."
        )

    label_mat = np.zeros((n_drug, n_dis), dtype=np.int8)
    label_mat[assoc["Drug"].values, assoc["Disease"].values] = 1

    print(f"RepoClin drug npy:    {drug_npy} shape={drug_x.shape}")
    print(f"RepoClin disease npy: {disease_npy} shape={disease_x.shape}")
    print(f"RepoClin positives:   {int(label_mat.sum())}")
    print(f"RepoClin matrix:      {n_drug} x {n_dis} = {n_drug * n_dis:,} pairs")
    return drug_x, disease_x, label_mat, assoc


# ============================================================
# Graph / model helpers copied from training logic
# ============================================================

def build_seen_only_heterograph(g_full, seen_drugs: np.ndarray, seen_diseases: np.ndarray, device: th.device):
    node_dict = {}
    for ntype in g_full.ntypes:
        if ntype == "drug":
            node_dict[ntype] = th.tensor(seen_drugs, dtype=th.long, device=g_full.device)
        elif ntype == "disease":
            node_dict[ntype] = th.tensor(seen_diseases, dtype=th.long, device=g_full.device)
        else:
            node_dict[ntype] = th.arange(g_full.num_nodes(ntype), dtype=th.long, device=g_full.device)
    return dgl.node_subgraph(g_full, node_dict, relabel_nodes=True, store_ids=True).to(device)


def make_dummy_similarity_graph(S_sub: th.Tensor):
    n = S_sub.shape[0]
    g = dgl.graph(([], []), num_nodes=n, device=S_sub.device)
    g.ndata["sim_feature"] = S_sub.float()
    return g


def get_sim_matrix_from_graph(sim_graph, device: th.device) -> th.Tensor:
    if "sim_feature" not in sim_graph.ndata:
        raise KeyError("similarity graph ndata['sim_feature'] not found")
    return sim_graph.ndata["sim_feature"].float().to(device)


def build_seen_similarity_graphs(full_dr_graph, full_di_graph, seen_drugs: np.ndarray, seen_diseases: np.ndarray, device):
    S_dr_full = get_sim_matrix_from_graph(full_dr_graph, device)
    S_di_full = get_sim_matrix_from_graph(full_di_graph, device)
    seen_dr_t = th.tensor(seen_drugs, dtype=th.long, device=device)
    seen_di_t = th.tensor(seen_diseases, dtype=th.long, device=device)
    S_dr_seen = S_dr_full.index_select(0, seen_dr_t).index_select(1, seen_dr_t)
    S_di_seen = S_di_full.index_select(0, seen_di_t).index_select(1, seen_di_t)
    return make_dummy_similarity_graph(S_dr_seen), make_dummy_similarity_graph(S_di_seen)


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
def get_final_node_embeddings(model, g_het, x_het, dr_graph, di_graph, device: th.device):
    model.eval()
    dummy_drug = th.zeros(1, dtype=th.long, device=device)
    dummy_dis = th.zeros(1, dtype=th.long, device=device)
    _, aux = model_forward(
        model, g_het, x_het, dr_graph, di_graph,
        dummy_drug, dummy_dis, return_aux=True,
    )
    if "final_node_emb" not in aux:
        raise KeyError("Model aux does not contain final_node_emb. Check model_update5.py forward output.")
    return aux["final_node_emb"]


def build_model(args, g_seen, n_drug_seen: int, n_dis_seen: int, device: th.device):
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
    return model


# ============================================================
# External double-cold pseudo transfer
# ============================================================

@th.no_grad()
def topk_pseudo_from_external_features(
    query_x: th.Tensor,
    seen_raw_x: th.Tensor,
    seen_final_emb: th.Tensor,
    k: int,
    temperature: float,
    chunk_size: int,
    device: th.device,
    return_details: bool = False,
    entity_type: str = "drug",
):
    """
    query_x:        [N_external, D_raw]
    seen_raw_x:     [N_seen, D_raw]
    seen_final_emb: [N_seen, D_hidden]
    return pseudo:  [N_external, D_hidden]
    """
    query_x = query_x.to(device).float()
    seen_raw_x = seen_raw_x.to(device).float()
    seen_final_emb = seen_final_emb.to(device).float()

    if query_x.shape[1] != seen_raw_x.shape[1]:
        raise ValueError(
            f"Raw feature dimension mismatch for {entity_type}: "
            f"external={query_x.shape[1]}, RepoApp seen={seen_raw_x.shape[1]}."
        )

    seen_norm = F.normalize(seen_raw_x, p=2, dim=1)
    k_eff = max(1, min(int(k), seen_norm.shape[0]))
    pseudo_list = []
    detail_rows = []

    for st in range(0, query_x.shape[0], int(chunk_size)):
        ed = min(st + int(chunk_size), query_x.shape[0])
        q = F.normalize(query_x[st:ed], p=2, dim=1)
        sim = th.mm(q, seen_norm.t())  # [B, N_seen]
        top_scores, top_idx = th.topk(sim, k=k_eff, dim=1, largest=True, sorted=True)
        weights = th.softmax(top_scores / float(temperature), dim=1)
        top_emb = seen_final_emb.index_select(0, top_idx.reshape(-1)).reshape(top_idx.shape[0], k_eff, -1)
        pseudo = th.sum(weights.unsqueeze(-1) * top_emb, dim=1)
        pseudo_list.append(pseudo)

        if return_details:
            top_idx_cpu = top_idx.detach().cpu().numpy()
            top_scores_cpu = top_scores.detach().cpu().numpy()
            weights_cpu = weights.detach().cpu().numpy()
            for bi in range(top_idx_cpu.shape[0]):
                eid = st + bi
                detail_rows.append({
                    "entity_type": entity_type,
                    "external_entity_id": int(eid),
                    "top_seen_local_ids": " ".join(map(str, top_idx_cpu[bi].astype(int).tolist())),
                    "cosine_scores": " ".join([f"{x:.8f}" for x in top_scores_cpu[bi].tolist()]),
                    "weights": " ".join([f"{x:.8f}" for x in weights_cpu[bi].tolist()]),
                })

    return th.cat(pseudo_list, dim=0), detail_rows


@th.no_grad()
def decode_pairs(pair_decoder, drug_h: th.Tensor, disease_h: th.Tensor, drug_ids: np.ndarray, disease_ids: np.ndarray,
                 batch_size: int, device: th.device) -> np.ndarray:
    scores = []
    drug_ids = np.asarray(drug_ids, dtype=np.int64)
    disease_ids = np.asarray(disease_ids, dtype=np.int64)
    for st in range(0, len(drug_ids), int(batch_size)):
        ed = min(st + int(batch_size), len(drug_ids))
        d = th.tensor(drug_ids[st:ed], dtype=th.long, device=device)
        s = th.tensor(disease_ids[st:ed], dtype=th.long, device=device)
        dh = drug_h.index_select(0, d)
        sh = disease_h.index_select(0, s)
        pair_feat = pair_decoder.pair_feature(dh, sh)
        logits = pair_decoder.mlp(pair_feat).squeeze(-1)
        scores.append(th.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0)


def build_eval_pairs(label_mat: np.ndarray, mode: str, max_eval_neg: int, seed: int):
    pos = np.argwhere(label_mat > 0)
    if mode == "all":
        all_drug, all_dis = np.indices(label_mat.shape)
        drug_ids = all_drug.reshape(-1)
        disease_ids = all_dis.reshape(-1)
        labels = label_mat.reshape(-1).astype(np.float32)
        return drug_ids, disease_ids, labels

    if mode == "sampled":
        neg = np.argwhere(label_mat <= 0)
        rng = np.random.default_rng(seed)
        if max_eval_neg is not None and int(max_eval_neg) > 0 and len(neg) > int(max_eval_neg):
            neg = neg[rng.choice(len(neg), size=int(max_eval_neg), replace=False)]
        pairs = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate([np.ones(len(pos), dtype=np.float32), np.zeros(len(neg), dtype=np.float32)])
        perm = rng.permutation(len(labels))
        pairs = pairs[perm]
        labels = labels[perm]
        return pairs[:, 0], pairs[:, 1], labels

    raise ValueError(f"Unsupported eval_pair_mode: {mode}")


def evaluate_scores(labels: np.ndarray, scores: np.ndarray):
    auc, aupr = get_metrics_auc(labels, scores)
    _, _, acc, f1, pre, rec, spe = get_metrics(labels, scores)
    return {
        "auc": float(auc),
        "aupr": float(aupr),
        "acc": float(acc),
        "f1": float(f1),
        "pre": float(pre),
        "rec": float(rec),
        "spe": float(spe),
    }


# ============================================================
# Main evaluation
# ============================================================

def get_fold_dirs(ckpt_root: str, folds: str):
    if folds.lower() == "all":
        out = []
        for name in sorted(os.listdir(ckpt_root)):
            if name.startswith("fold") and os.path.isdir(os.path.join(ckpt_root, name)):
                try:
                    fid = int(name.replace("fold", ""))
                except ValueError:
                    continue
                out.append((fid, os.path.join(ckpt_root, name)))
        return sorted(out, key=lambda x: x[0])
    ids = [int(x.strip()) for x in folds.split(",") if x.strip()]
    return [(fid, os.path.join(ckpt_root, f"fold{fid}")) for fid in ids]


def main():
    args = build_parser().parse_args()
    if args.sim_no_diffusion:
        args.sim_use_diffusion = False

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = open(os.path.join(args.save_dir, "eval_repoapp_to_repoclin_external_double_cold.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    print(args)
    device = th.device(f"cuda:{args.device_id}" if th.cuda.is_available() else "cpu")
    print("Evaluating on", device)

    # Load RepoClin external data.
    repoclin_drug_x_np, repoclin_disease_x_np, repoclin_label_mat, assoc_df = load_repoclin_data(args)
    repoclin_drug_x = th.tensor(repoclin_drug_x_np, dtype=th.float32, device=device)
    repoclin_disease_x = th.tensor(repoclin_disease_x_np, dtype=th.float32, device=device)

    eval_drug_ids, eval_disease_ids, eval_labels = build_eval_pairs(
        repoclin_label_mat,
        mode=args.eval_pair_mode,
        max_eval_neg=args.max_eval_neg,
        seed=args.seed,
    )
    print(f"Eval mode: {args.eval_pair_mode}")
    print(f"Eval pairs: {len(eval_labels):,}; positives={int(np.sum(eval_labels > 0.5)):,}; negatives={int(np.sum(eval_labels <= 0.5)):,}")

    # Load RepoApp graph/similarity once.
    repoapp_base_dir = args.repoapp_base_dir or os.path.join("./dataset", args.repoapp_dataset)
    g_full = safe_load_heterograph(
        args.repoapp_dataset,
        device=device,
        base_dir=repoapp_base_dir,
        device_id=int(args.device_id),
        feature_mode=args.feature_mode,
    ).to(device)
    x_full = get_feature_dict(g_full)
    raw_repoapp_drug_all = x_full["drug"].detach().to(device)
    raw_repoapp_disease_all = x_full["disease"].detach().to(device)
    seen_drugs_all = np.arange(g_full.num_nodes("drug"), dtype=np.int64)
    seen_diseases_all = np.arange(g_full.num_nodes("disease"), dtype=np.int64)
    print(f"RepoApp source entities used for pseudo transfer: drugs={len(seen_drugs_all):,}, diseases={len(seen_diseases_all):,}")

    full_dr_graph, full_di_graph = prepare_similarity_graphs(
        dataset=args.repoapp_dataset,
        base_dir=repoapp_base_dir,
        K=args.sim_topk,
        device=device,
        make_undirected=True,
    )
    full_dr_graph = full_dr_graph.to(device)
    full_di_graph = full_di_graph.to(device)

    fold_dirs = get_fold_dirs(args.repoapp_ckpt_root, args.folds)
    if len(fold_dirs) == 0:
        raise RuntimeError(f"No fold dirs found under {args.repoapp_ckpt_root}")
    print("Fold dirs:", fold_dirs)

    summary_rows = []
    fold_score_list = []

    for fold_id, fold_dir in fold_dirs:
        t0 = time.time()
        print("\n" + "=" * 80)
        print(f"External double-cold pseudo evaluation | Fold {fold_id}")
        print("=" * 80)

        ckpt_path = os.path.join(fold_dir, "best_model_state.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        # External RepoClin testing: use the entire RepoApp graph as the seen source graph.
        # RepoClin drugs and RepoClin diseases are both external cold entities, so they are
        # represented by pseudo embeddings transferred from all RepoApp drugs/diseases.
        seen_drugs = seen_drugs_all
        seen_diseases = seen_diseases_all
        print(f"RepoApp seen source drugs: {len(seen_drugs):,}; seen source diseases: {len(seen_diseases):,}")

        g_seen = g_full
        x_seen = {k: v.to(device) for k, v in x_full.items()}
        dr_seen_graph, di_seen_graph = full_dr_graph, full_di_graph

        model = build_model(
            args,
            g_seen=g_seen,
            n_drug_seen=dr_seen_graph.num_nodes(),
            n_dis_seen=di_seen_graph.num_nodes(),
            device=device,
        )
        model.set_similarity_graphs(dr_seen_graph, di_seen_graph, device=device)

        state = th.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print("[Warning] load_state_dict strict=False")
            print("  missing keys:", missing[:20], "..." if len(missing) > 20 else "")
            print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")
        model.eval()

        final_node_emb = get_final_node_embeddings(model, g_seen, x_seen, dr_seen_graph, di_seen_graph, device)
        if "drug" not in final_node_emb or "disease" not in final_node_emb:
            raise KeyError("final_node_emb must contain drug and disease embeddings.")

        # Raw seen features are indexed from the full RepoApp graph by original ids.
        seen_drug_raw = raw_repoapp_drug_all.index_select(0, th.tensor(seen_drugs, dtype=th.long, device=device))
        seen_disease_raw = raw_repoapp_disease_all.index_select(0, th.tensor(seen_diseases, dtype=th.long, device=device))

        repoclin_drug_h, drug_detail_rows = topk_pseudo_from_external_features(
            query_x=repoclin_drug_x,
            seen_raw_x=seen_drug_raw,
            seen_final_emb=final_node_emb["drug"],
            k=args.pseudo_k,
            temperature=args.pseudo_temperature,
            chunk_size=args.retrieval_chunk_size,
            device=device,
            return_details=args.save_pseudo_details,
            entity_type="drug",
        )
        repoclin_disease_h, disease_detail_rows = topk_pseudo_from_external_features(
            query_x=repoclin_disease_x,
            seen_raw_x=seen_disease_raw,
            seen_final_emb=final_node_emb["disease"],
            k=args.pseudo_k,
            temperature=args.pseudo_temperature,
            chunk_size=args.retrieval_chunk_size,
            device=device,
            return_details=args.save_pseudo_details,
            entity_type="disease",
        )

        scores = decode_pairs(
            model.pair_decoder,
            repoclin_drug_h,
            repoclin_disease_h,
            drug_ids=eval_drug_ids,
            disease_ids=eval_disease_ids,
            batch_size=args.eval_batch_size,
            device=device,
        )
        metrics = evaluate_scores(eval_labels, scores)
        fold_score_list.append(scores)

        row = {
            "fold": fold_id,
            "ckpt_path": ckpt_path,
            "pseudo_k": args.pseudo_k,
            "pseudo_temperature": args.pseudo_temperature,
            "eval_pair_mode": args.eval_pair_mode,
            "num_eval_pairs": int(len(eval_labels)),
            "num_pos": int(np.sum(eval_labels > 0.5)),
            "num_neg": int(np.sum(eval_labels <= 0.5)),
            **metrics,
            "time_sec": float(time.time() - t0),
        }
        summary_rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

        fold_out_dir = os.path.join(args.save_dir, f"fold{fold_id}")
        os.makedirs(fold_out_dir, exist_ok=True)
        with open(os.path.join(fold_out_dir, "repoclin_external_double_cold_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

        if args.save_fold_predictions:
            pd.DataFrame({
                "Drug": eval_drug_ids.astype(int),
                "Disease": eval_disease_ids.astype(int),
                "label": eval_labels.astype(int),
                "score": scores.astype(float),
            }).to_csv(os.path.join(fold_out_dir, "repoclin_predictions.csv"), index=False)

        if args.save_pseudo_details:
            pd.DataFrame(drug_detail_rows).to_csv(os.path.join(fold_out_dir, "repoclin_drug_pseudo_neighbors.csv"), index=False)
            pd.DataFrame(disease_detail_rows).to_csv(os.path.join(fold_out_dir, "repoclin_disease_pseudo_neighbors.csv"), index=False)

    # Save per-fold summary.
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.save_dir, "repoclin_external_double_cold_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved per-fold summary: {summary_path}")

    # Mean/std across folds.
    mean_std_rows = []
    for key in ["auc", "aupr", "acc", "f1", "pre", "rec", "spe"]:
        vals = summary_df[key].astype(float).values
        mean_std_rows.append({
            "metric": key,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "mean_std": f"{np.mean(vals):.5f} ± {(np.std(vals, ddof=1) if len(vals) > 1 else 0.0):.5f}",
        })
    mean_std_df = pd.DataFrame(mean_std_rows)
    mean_std_path = os.path.join(args.save_dir, "repoclin_external_double_cold_mean_std.csv")
    mean_std_df.to_csv(mean_std_path, index=False)
    print(f"Saved mean/std: {mean_std_path}")
    print(mean_std_df)

    # Ensemble scores: average scores from all fold models.
    if len(fold_score_list) > 0:
        ensemble_scores = np.mean(np.stack(fold_score_list, axis=0), axis=0)
        ensemble_metrics = evaluate_scores(eval_labels, ensemble_scores)
        ensemble_json_path = os.path.join(args.save_dir, "repoclin_external_double_cold_ensemble_metrics.json")
        with open(ensemble_json_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved ensemble metrics: {ensemble_json_path}")
        print("Ensemble metrics:", json.dumps(ensemble_metrics, indent=2, ensure_ascii=False))

        if args.save_ensemble_predictions:
            ensemble_pred_path = os.path.join(args.save_dir, "repoclin_external_double_cold_ensemble_predictions.csv")
            pd.DataFrame({
                "Drug": eval_drug_ids.astype(int),
                "Disease": eval_disease_ids.astype(int),
                "label": eval_labels.astype(int),
                "score": ensemble_scores.astype(float),
            }).to_csv(ensemble_pred_path, index=False)
            print(f"Saved ensemble predictions: {ensemble_pred_path}")

    log_file.close()


def build_parser():
    parser = argparse.ArgumentParser()

    # RepoApp trained model / data.
    parser.add_argument("--repoapp_dataset", default="RepoApp")
    parser.add_argument("--repoapp_base_dir", default=None)
    parser.add_argument("--repoapp_ckpt_root", required=True,
                        help="Directory containing fold1/.../best_model_state.pth")
    parser.add_argument("--folds", default="all", help="all or comma-separated fold ids, e.g. 1,2,3")

    # RepoClin external test data.
    parser.add_argument("--repoclin_dir", required=True)
    parser.add_argument("--repoclin_drug_npy", default=None)
    parser.add_argument("--repoclin_disease_npy", default=None)
    parser.add_argument("--repoclin_assoc_csv", required=True)

    # Output.
    parser.add_argument("--save_dir", default="result_repoapp_to_repoclin_external_double_cold")
    parser.add_argument("--save_fold_predictions", action="store_true", default=False)
    parser.add_argument("--save_ensemble_predictions", action="store_true", default=True)
    parser.add_argument("--save_pseudo_details", action="store_true", default=False)

    # Device / seed.
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_mode", choices=["llm", "random"], default="llm")

    # Evaluation pair selection.
    parser.add_argument("--eval_pair_mode", choices=["all", "sampled"], default="all")
    parser.add_argument("--max_eval_neg", type=int, default=100000,
                        help="Only used when --eval_pair_mode sampled.")
    parser.add_argument("--eval_batch_size", type=int, default=200000)
    parser.add_argument("--retrieval_chunk_size", type=int, default=1024)

    # External pseudo transfer. No drug_cold / disease_cold mode here.
    parser.add_argument("--pseudo_k", type=int, default=10)
    parser.add_argument("--pseudo_temperature", type=float, default=0.1)

    # Model architecture: must match training.
    parser.add_argument("--hidden_feats", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_sim_layers", type=int, default=2)
    parser.add_argument("--query_layers", type=str, default="1")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pair_hidden", type=int, default=128)
    parser.add_argument("--pair_mode", choices=["rotate", "absdiff"], default="rotate")
    parser.add_argument("--layer_pooling", choices=["last", "mean", "dream", "attn"], default="attn")

    # Similarity / query, needed to reconstruct model and seen similarity graph.
    parser.add_argument("--sim_topk", type=int, default=5)
    parser.add_argument("--sim_use_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_no_diffusion", action="store_true", default=False)
    parser.add_argument("--sim_diffusion_alpha", type=float, default=0.15)
    parser.add_argument("--sim_diffusion_steps", type=int, default=3)
    parser.add_argument("--sim_use_diffused_adj_for_gcn", action="store_true", default=False)
    parser.add_argument("--query_gamma_init", type=float, default=0.05)
    parser.add_argument("--cl_temperature", type=float, default=0.2)
    parser.add_argument("--cl_sample_size", type=int, default=0)

    return parser


if __name__ == "__main__":
    main()
