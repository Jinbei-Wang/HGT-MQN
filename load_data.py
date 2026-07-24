"""
Unified load_data.py
"""

import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import dgl
from typing import Dict, Tuple, Optional
from collections import defaultdict

def resolve_device(device: Optional[th.device] = None, device_id: Optional[str] = "0") -> th.device:
    # 优先用外部传入的 device
    if device is not None:
        return device
    # 否则尝试用 device_id 走 GPU
    if device_id is not None and str(device_id) != "" and th.cuda.is_available():
        return th.device(f"cuda:{device_id}")
    return th.device("cpu")

#-----------------读取原始的baseline相似度分数数据文件，并得到TOPK------------
@th.no_grad()
def load_baseline_sim_csv(path: str, device: th.device, dtype=th.float32) -> th.Tensor:
    """读取 baseline 相似度矩阵 CSV -> [N,N] tensor"""
    df = pd.read_csv(path, header=None).fillna(0.0)
    return th.tensor(df.to_numpy(), dtype=dtype, device=device)

#-----------------读取llm生成的npy数据文件，利用余弦相似度构建TOPK列表------------
@th.no_grad()
def load_llm_emb_npy(path: str, device: th.device, dtype=th.float32) -> th.Tensor:
    """读取对齐后的 LLM embedding npy -> [N,d] tensor"""
    X = np.load(path)
    return th.from_numpy(X).to(device=device, dtype=dtype)

@th.no_grad()
def knn_from_llm_emb(
    llm_emb: th.Tensor,
    K: int = 15,
    chunk_size: int = 4096,
) -> th.Tensor:
    """
    llm_emb: [N,d] -> knn_het: [N,K]
    余弦相似度 TopK，chunk 计算避免 [N,N] 爆内存
    """
    assert llm_emb.dim() == 2
    N = llm_emb.size(0)
    K = min(K, N - 1)

    X = F.normalize(llm_emb, p=2, dim=1)  # [N,d]
    idx_all = th.empty((N, K), dtype=th.long, device=X.device)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        Xb = X[start:end]            # [b,d]
        sim = Xb @ X.t()             # [b,N]

        # 排除 self
        r = th.arange(end - start, device=X.device)
        sim[r, start + r] = -float("inf")

        _, idx = th.topk(sim, k=K, dim=1, largest=True, sorted=True)
        idx_all[start:end] = idx

    return idx_all

# =========================================================
# Part B. Sim-GT 图构建：dr_graph / di_graph
# =========================================================

@th.no_grad()
def _build_sim_Kdataset(sim_matrix: th.Tensor, K: int, make_undirected: bool = True) -> dgl.DGLGraph:
    assert sim_matrix.dim() == 2 and sim_matrix.size(0) == sim_matrix.size(1)

    out_device = sim_matrix.device
    N = sim_matrix.size(0)
    K = min(K, N - 1)

    S = sim_matrix.clone()
    S.fill_diagonal_(-float("inf"))
    vals, idx = th.topk(S, k=K, dim=1, largest=True, sorted=True)

    src = th.arange(N, device=out_device).unsqueeze(1).repeat(1, K).reshape(-1)
    dst = idx.reshape(-1)
    w   = vals.reshape(-1)

    # ---- DGL transforms: 必须 CPU ----
    src_cpu, dst_cpu, w_cpu = src.cpu(), dst.cpu(), w.cpu()
    g = dgl.graph((src_cpu, dst_cpu), num_nodes=N, device=th.device("cpu"))
    g.edata["sim"] = w_cpu
    g.edata["w"] = w_cpu

    if make_undirected:
        g = dgl.add_reverse_edges(g, copy_edata=True)
        #g = dgl.to_simple(g)

    # ---- 回到 GPU（你已经验证 DGL 支持 CUDA）----
    if out_device.type == "cuda":
        g = g.to(out_device)

    return g


@th.no_grad()
def prepare_similarity_graphs(
    *,
    dataset: str,
    base_dir: Optional[str],
    K: int,
    device: th.device,
    dtype=th.float32,
    make_undirected: bool = True,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """
    返回 (dr_graph, di_graph) 供你的 sim_gt(GraphTransformer) 使用：
      - graph.ndata["sim_feature"] 必须存在
      - graph.edata["sim"] 作为边权 bias（如果 GraphTransformer 用得上）
    """
    if base_dir is None:
        base_dir = os.path.join("./dataset", dataset)

    drug_sim_csv = os.path.join(base_dir, "drug_drug_baseline.csv")
    dis_sim_csv = os.path.join(base_dir, "disease_disease_baseline.csv")
    # 读取
    S_dr = load_baseline_sim_csv(drug_sim_csv, device=device, dtype=dtype)
    S_di = load_baseline_sim_csv(dis_sim_csv, device=device, dtype=dtype)

    dr_graph = _build_sim_Kdataset(S_dr, K=K, make_undirected=make_undirected)
    di_graph = _build_sim_Kdataset(S_di, K=K, make_undirected=make_undirected)

    dr_graph.ndata["sim_feature"] = S_dr
    di_graph.ndata["sim_feature"] = S_di
    return dr_graph, di_graph


# =========================================================
# Part C. REDDA 异构图加载（原 load_data.py 合并过来）
# =========================================================

def load(
        dataset: str,
        device: Optional[th.device] = None,
        device_id: Optional[str] = "0",
        **kwargs
):
    """
    兼容旧调用：load(dataset)
    新实验推荐用：load(dataset, feature_mode="llm", llm_feat_dim=768, base_dir=..., ...)
    """
    device = resolve_device(device=device, device_id=device_id)
    if dataset == "Bdataset":
        return load_Bdataset(**kwargs)
    if dataset == "Cdataset":
        return load_Cdataset(**kwargs)
    if dataset == "RepoApp":
        return load_RepoApp(**kwargs)
    if dataset == "Kdataset":
        return load_Kdataset(**kwargs)
    raise ValueError("Unsupported dataset. Choose from Bdataset/Cdataset/Kdataset/RepoApp.")

def knn_idx_to_edges(knn_idx: th.Tensor, *, symmetrize: bool = False) -> Tuple[th.Tensor, th.Tensor]:
    """
    knn_idx: [N,K] long -> (src,dst) each [E]
    symmetrize=True: add reverse edges and unique
    我们之前写的算法因为输出的topk列表，在这里要转成索引供之后Kdataset调用
    """
    assert knn_idx.dim() == 2
    N, K = knn_idx.shape
    device = knn_idx.device

    src = th.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)
    dst = knn_idx.reshape(-1)

    if not symmetrize:
        return src, dst

    src2 = th.cat([src, dst], dim=0)
    dst2 = th.cat([dst, src], dim=0)
    # unique edges
    # encode pair -> unique -> decode
    key = src2 * N + dst2
    uniq = th.unique(key)
    src_u = uniq // N
    dst_u = uniq % N
    return src_u, dst_u

def load_Kdataset(
    *,
    base_dir: str = "./dataset/Kdataset",
    topk_bin: int = 15,
    llm_feat_dim: int = 3072,
    other_feat_dim: int = 128,
    chunk_size: int = 4096,
    knn_symmetrize: bool = False,
    device: th.device = th.device("cpu"),
    feature_mode: str = "llm",
    **kwargs
)-> dgl.DGLHeteroGraph:
    """
        - drug-drug / disease-disease：用 LLM embedding cosine TopK 构边
        - 节点特征：
            drug/disease: [N, llm_feat_dim]
            others: [N_other, other_feat_dim] 全0
            此外，Kdataset的异构图有相似性药物的边，如果后期更改，则再进行修改
    """
    # ---- load LLM embeddings (on device)
    drug_path = os.path.join(base_dir, "drug_LLM_emb.npy")
    dis_path = os.path.join(base_dir, "disease_LLM_emb.npy")
    if not (os.path.exists(drug_path) and os.path.exists(dis_path)):
        raise FileNotFoundError(f"LLM emb not found under {base_dir}")

    dr_llm_feature = load_llm_emb_npy(drug_path, device=device, dtype=th.float32)
    di_llm_feature = load_llm_emb_npy(dis_path, device=device, dtype=th.float32)
    assert dr_llm_feature.size(1) == llm_feat_dim and di_llm_feature.size(1) == llm_feat_dim

    knn_dr = knn_from_llm_emb(dr_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_dr,K]
    knn_di = knn_from_llm_emb(di_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_di,K]
    dr_src, dr_dst = knn_idx_to_edges(knn_dr, symmetrize=knn_symmetrize) # construct drug sim edges
    di_src, di_dst = knn_idx_to_edges(knn_di, symmetrize=knn_symmetrize) # construct dis sim edges


    protein_protein = pd.read_csv(os.path.join(base_dir, "interactions/protein_protein.csv"))
    gene_gene = pd.read_csv(os.path.join(base_dir, "interactions/gene_gene.csv"))
    pathway_pathway = pd.read_csv(os.path.join(base_dir, "interactions/pathway_pathway.csv"))

    drug_protein = pd.read_csv(os.path.join(base_dir, "associations/drug_protein.csv"))
    protein_gene = pd.read_csv(os.path.join(base_dir, "associations/protein_gene.csv"))
    gene_pathway = pd.read_csv(os.path.join(base_dir, "associations/gene_pathway.csv"))
    pathway_disease = pd.read_csv(os.path.join(base_dir, "associations/pathway_disease.csv"))
    drug_disease = pd.read_csv(os.path.join(base_dir, "associations/Kdataset.csv"))

    def t(x):
        return th.tensor(x, device=device, dtype=th.long)

    graph_data = {
        # LLM-KNN edges
        ("drug", "drug_drug", "drug"): (dr_src, dr_dst),
        ("disease", "disease_disease", "disease"): (di_src, di_dst),

        # other relations (same as before)
        ("drug", "drug_protein", "protein"): (t(drug_protein["Drug"].values), t(drug_protein["Protein"].values)),
        ("protein", "protein_drug", "drug"): (t(drug_protein["Protein"].values), t(drug_protein["Drug"].values)),

        ("protein", "protein_protein", "protein"): (t(protein_protein["Protein1"].values), t(protein_protein["Protein2"].values)),

        ("protein", "protein_gene", "gene"): (t(protein_gene["Protein"].values), t(protein_gene["Gene"].values)),
        ("gene", "gene_protein", "protein"): (t(protein_gene["Gene"].values), t(protein_gene["Protein"].values)),

        ("gene", "gene_gene", "gene"): (t(gene_gene["Gene1"].values), t(gene_gene["Gene2"].values)),

        ("gene", "gene_pathway", "pathway"): (t(gene_pathway["Gene"].values), t(gene_pathway["Pathway"].values)),
        ("pathway", "pathway_gene", "gene"): (t(gene_pathway["Pathway"].values), t(gene_pathway["Gene"].values)),

        ("pathway", "pathway_pathway", "pathway"): (t(pathway_pathway["Pathway1"].values),t(pathway_pathway["Pathway2"].values)),

        ("pathway", "pathway_disease", "disease"): (t(pathway_disease["Pathway"].values),t(pathway_disease["Disease"].values)),
        ("disease", "disease_pathway", "pathway"): (t(pathway_disease["Disease"].values),t(pathway_disease["Pathway"].values)),

        ("drug", "drug_disease", "disease"): (t(drug_disease["Drug"].values), t(drug_disease["Disease"].values)),
        ("disease", "disease_drug", "drug"): (t(drug_disease["Disease"].values), t(drug_disease["Drug"].values)),}

    g = dgl.heterograph(graph_data, device=device)

    assert g.num_nodes("drug") == dr_llm_feature.size(0)
    assert g.num_nodes("disease") == di_llm_feature.size(0)

    g.nodes["drug"].data["h"] = dr_llm_feature.to(th.float32)
    g.nodes["disease"].data["h"] = di_llm_feature.to(th.float32)
    # 其他类型128 维度
    # for ntype in g.ntypes:
    #     if ntype in ("drug", "disease"):
    #         continue
    #     g.nodes[ntype].data["h"] = th.zeros((g.num_nodes(ntype), other_feat_dim), dtype=th.float32, device=g.device)
    #
    # return g
    if feature_mode == "llm":
        g.nodes["drug"].data["h"] = dr_llm_feature.to(th.float32)
        g.nodes["disease"].data["h"] = di_llm_feature.to(th.float32)
    elif feature_mode == "random":
        g.nodes["drug"].data["h"] = th.randn((g.num_nodes("drug"), llm_feat_dim), dtype=th.float32, device=g.device, )
        g.nodes["disease"].data["h"] = th.randn((g.num_nodes("disease"), llm_feat_dim), dtype=th.float32, device=g.device, )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    # 其他类型保持不变
    for ntype in g.ntypes:
        if ntype in ("drug", "disease"):
            continue
        g.nodes[ntype].data["h"] = th.zeros((g.num_nodes(ntype), other_feat_dim), dtype=th.float32, device=g.device)

    return g

def load_Bdataset(
    *,
    base_dir: str = "./dataset/Bdataset",
    llm_feat_dim: int = 1036,
    topk_bin: int = 15,
    other_feat_dim: int = 128,
    chunk_size: int = 4096,
    knn_symmetrize: bool = False,
    device: th.device = th.device("cpu"),
    feature_mode: str = "llm",
    **kwargs
):
    drug_path = os.path.join(base_dir, "drug_LLM_emb.npy")
    dis_path = os.path.join(base_dir, "disease_LLM_emb.npy")
    if not (os.path.exists(drug_path) and os.path.exists(dis_path)):
        raise FileNotFoundError(f"LLM emb not found under {base_dir}")

    dr_llm_feature = load_llm_emb_npy(drug_path, device=device, dtype=th.float32)
    di_llm_feature = load_llm_emb_npy(dis_path, device=device, dtype=th.float32)
    assert dr_llm_feature.size(1) == llm_feat_dim and di_llm_feature.size(1) == llm_feat_dim

    knn_dr = knn_from_llm_emb(dr_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_dr,K]
    knn_di = knn_from_llm_emb(di_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_di,K]
    dr_src, dr_dst = knn_idx_to_edges(knn_dr, symmetrize=knn_symmetrize)  # construct drug sim edges
    di_src, di_dst = knn_idx_to_edges(knn_di, symmetrize=knn_symmetrize)  # construct dis sim edges

    protein_protein = pd.read_csv(os.path.join(base_dir, "interactions/protein_protein.csv"))
    drug_protein = pd.read_csv(os.path.join(base_dir, "associations/drug_protein.csv"))
    protein_disease = pd.read_csv(os.path.join(base_dir,"associations/protein_disease.csv" ))
    drug_disease = pd.read_csv(os.path.join(base_dir, "associations/Bdataset.csv"))

    graph_data = {
        ("drug", "drug_drug", "drug"): (dr_src, dr_dst),
        ("disease", "disease_disease", "disease"): (di_src, di_dst),
        ("drug", "drug_protein", "protein"): (th.tensor(drug_protein["Drug"].values), th.tensor(drug_protein["Protein"].values)),
        ("protein", "protein_drug", "drug"): (th.tensor(drug_protein["Protein"].values), th.tensor(drug_protein["Drug"].values)),
        ("disease", "disease_protein", "protein"): (th.tensor(protein_disease["Disease"].values), th.tensor(protein_disease["Protein"].values)),
        ("protein", "protein_disease", "disease"): (th.tensor(protein_disease["Protein"].values),th.tensor(protein_disease["Disease"].values)),
        ("protein", "protein_protein", "protein"): (th.tensor(protein_protein["Protein1"].values), th.tensor(protein_protein["Protein2"].values)),
        ("drug", "drug_disease", "disease"): (th.tensor(drug_disease["Drug"].values), th.tensor(drug_disease["Disease"].values)),
        ("disease", "disease_drug", "drug"): (th.tensor(drug_disease["Disease"].values), th.tensor(drug_disease["Drug"].values)),
    }
    g = dgl.heterograph(graph_data)

    assert g.num_nodes("drug") == dr_llm_feature.size(0)
    assert g.num_nodes("disease") == di_llm_feature.size(0)

    if feature_mode == "llm":
        g.nodes["drug"].data["h"] = dr_llm_feature.to(th.float32)
        g.nodes["disease"].data["h"] = di_llm_feature.to(th.float32)
    elif feature_mode == "random":
        g.nodes["drug"].data["h"] = th.randn((g.num_nodes("drug"), llm_feat_dim), dtype=th.float32, device=g.device, )
        g.nodes["disease"].data["h"] = th.randn((g.num_nodes("disease"), llm_feat_dim), dtype=th.float32, device=g.device, )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    # 其他类型128 维度
    for ntype in g.ntypes:
        if ntype in ("drug", "disease"):
            continue
        g.nodes[ntype].data["h"] = th.zeros((g.num_nodes(ntype), other_feat_dim), dtype=th.float32, device=g.device)

    return g
    # baseline 特征（兼容）
    # drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes("drug"), g.num_nodes("disease")))))
    # dis_feature = np.hstack((np.zeros((g.num_nodes("disease"), g.num_nodes("drug"))), disease_sim))
    # g.nodes["drug"].data["h"] = th.from_numpy(drug_feature).to(th.float32)
    # g.nodes["disease"].data["h"] = th.from_numpy(dis_feature).to(th.float32)
    # g.nodes["protein"].data["h"] = th.zeros((g.num_nodes("protein"), drug_feature.shape[1])).to(th.float32)


def load_Cdataset(
    *,
    base_dir: str = "./dataset/Cdataset",
    llm_feat_dim: int = 3072,
    topk_bin: int = 15,
    other_feat_dim: int = 128,
    chunk_size: int = 4096,
    knn_symmetrize: bool = False,
    device: th.device = th.device("cpu"),
    feature_mode: str = "llm",
    **kwargs
):
    drug_path = os.path.join(base_dir, "drug_LLM_emb.npy")
    dis_path = os.path.join(base_dir, "disease_LLM_emb.npy")
    if not (os.path.exists(drug_path) and os.path.exists(dis_path)):
        raise FileNotFoundError(f"LLM emb not found under {base_dir}")

    dr_llm_feature = load_llm_emb_npy(drug_path, device=device, dtype=th.float32)
    di_llm_feature = load_llm_emb_npy(dis_path, device=device, dtype=th.float32)
    assert dr_llm_feature.size(1) == llm_feat_dim and di_llm_feature.size(1) == llm_feat_dim

    knn_dr = knn_from_llm_emb(dr_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_dr,K]
    knn_di = knn_from_llm_emb(di_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_di,K]
    dr_src, dr_dst = knn_idx_to_edges(knn_dr, symmetrize=knn_symmetrize)  # construct drug sim edges
    di_src, di_dst = knn_idx_to_edges(knn_di, symmetrize=knn_symmetrize)  # construct dis sim edges

    protein_protein = pd.read_csv(os.path.join(base_dir, "interactions/protein_protein.csv"))
    drug_protein = pd.read_csv(os.path.join(base_dir, "associations/drug_protein.csv"))
    protein_disease = pd.read_csv(os.path.join(base_dir, "associations/protein_disease.csv"))
    drug_disease = pd.read_csv(os.path.join(base_dir, "associations/Cdataset.csv"))

    graph_data = {
        ("drug", "drug_drug", "drug"): (dr_src, dr_dst),
        ("disease", "disease_disease", "disease"): (di_src, di_dst),
        ("drug", "drug_protein", "protein"): (th.tensor(drug_protein["Drug"].values), th.tensor(drug_protein["Protein"].values)),
        ("protein", "protein_drug", "drug"): (th.tensor(drug_protein["Protein"].values), th.tensor(drug_protein["Drug"].values)),
        ("disease", "disease_protein", "protein"): (th.tensor(protein_disease["Disease"].values),th.tensor(protein_disease["Protein"].values)),
        ("protein", "protein_disease", "disease"): (th.tensor(protein_disease["Protein"].values),th.tensor(protein_disease["Disease"].values)),
        ("protein", "protein_protein", "protein"): (th.tensor(protein_protein["Protein1"].values), th.tensor(protein_protein["Protein2"].values)),
        ("drug", "drug_disease", "disease"): (th.tensor(drug_disease["Drug"].values), th.tensor(drug_disease["Disease"].values)),
        ("disease", "disease_drug", "drug"): (th.tensor(drug_disease["Disease"].values), th.tensor(drug_disease["Drug"].values)),
    }
    g = dgl.heterograph(graph_data)

    assert g.num_nodes("drug") == dr_llm_feature.size(0)
    assert g.num_nodes("disease") == di_llm_feature.size(0)

    if feature_mode == "llm":
        g.nodes["drug"].data["h"] = dr_llm_feature.to(th.float32)
        g.nodes["disease"].data["h"] = di_llm_feature.to(th.float32)
    elif feature_mode == "random":
        g.nodes["drug"].data["h"] = th.randn((g.num_nodes("drug"), llm_feat_dim), dtype=th.float32, device=g.device, )
        g.nodes["disease"].data["h"] = th.randn((g.num_nodes("disease"), llm_feat_dim), dtype=th.float32, device=g.device, )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    # 其他类型128 维度
    for ntype in g.ntypes:
        if ntype in ("drug", "disease"):
            continue
        g.nodes[ntype].data["h"] = th.zeros((g.num_nodes(ntype), other_feat_dim), dtype=th.float32, device=g.device)

    return g

def load_RepoApp(
    *,
    base_dir: str = "./dataset/RepoApp",
    llm_feat_dim: int = 3072,
    topk_bin: int = 15,
    other_feat_dim: int = 128,
    chunk_size: int = 4096,
    knn_symmetrize: bool = False,
    device: th.device = th.device("cpu"),
    feature_mode: str = "llm",
    **kwargs
):
    drug_path = os.path.join(base_dir, "drug_embedding.npy")
    dis_path = os.path.join(base_dir, "disease_embedding.npy")
    if not (os.path.exists(drug_path) and os.path.exists(dis_path)):
        raise FileNotFoundError(f"LLM emb not found under {base_dir}")

    dr_llm_feature = load_llm_emb_npy(drug_path, device=device, dtype=th.float32)
    di_llm_feature = load_llm_emb_npy(dis_path, device=device, dtype=th.float32)
    assert dr_llm_feature.size(1) == llm_feat_dim and di_llm_feature.size(1) == llm_feat_dim

    knn_dr = knn_from_llm_emb(dr_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_dr,K]
    knn_di = knn_from_llm_emb(di_llm_feature, K=topk_bin, chunk_size=chunk_size)  # [N_di,K]
    dr_src, dr_dst = knn_idx_to_edges(knn_dr, symmetrize=knn_symmetrize)  # construct drug sim edges
    di_src, di_dst = knn_idx_to_edges(knn_di, symmetrize=knn_symmetrize)  # construct dis sim edges

    protein_protein = pd.read_csv(os.path.join(base_dir, "interactions/protein_protein.csv"))
    drug_protein = pd.read_csv(os.path.join(base_dir, "associations/drug_protein.csv"))
    protein_disease = pd.read_csv(os.path.join(base_dir, "associations/protein_disease.csv"))
    drug_disease = pd.read_csv(os.path.join(base_dir, "associations/RepoApp.csv"))

    graph_data = {
        ("drug", "drug_drug", "drug"): (dr_src, dr_dst),
        ("disease", "disease_disease", "disease"): (di_src, di_dst),
        ("drug", "drug_protein", "protein"): (th.tensor(drug_protein["Drug"].values), th.tensor(drug_protein["Protein"].values)),
        ("protein", "protein_drug", "drug"): (th.tensor(drug_protein["Protein"].values), th.tensor(drug_protein["Drug"].values)),
        ("disease", "disease_protein", "protein"): (th.tensor(protein_disease["Disease"].values),th.tensor(protein_disease["Protein"].values)),
        ("protein", "protein_disease", "disease"): (th.tensor(protein_disease["Protein"].values),th.tensor(protein_disease["Disease"].values)),
        ("protein", "protein_protein", "protein"): (th.tensor(protein_protein["Protein1"].values), th.tensor(protein_protein["Protein2"].values)),
        ("drug", "drug_disease", "disease"): (th.tensor(drug_disease["Drug"].values), th.tensor(drug_disease["Disease"].values)),
        ("disease", "disease_drug", "drug"): (th.tensor(drug_disease["Disease"].values), th.tensor(drug_disease["Drug"].values)),
    }
    g = dgl.heterograph(graph_data)

    assert g.num_nodes("drug") == dr_llm_feature.size(0)
    assert g.num_nodes("disease") == di_llm_feature.size(0)

    if feature_mode == "llm":
        g.nodes["drug"].data["h"] = dr_llm_feature.to(th.float32)
        g.nodes["disease"].data["h"] = di_llm_feature.to(th.float32)
    elif feature_mode == "random":
        g.nodes["drug"].data["h"] = th.randn((g.num_nodes("drug"), llm_feat_dim), dtype=th.float32, device=g.device, )
        g.nodes["disease"].data["h"] = th.randn((g.num_nodes("disease"), llm_feat_dim), dtype=th.float32, device=g.device, )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    # 其他类型128 维度
    for ntype in g.ntypes:
        if ntype in ("drug", "disease"):
            continue
        g.nodes[ntype].data["h"] = th.zeros((g.num_nodes(ntype), other_feat_dim), dtype=th.float32, device=g.device)

    return g

def remove_graph(g, test_id):
    """
    Remove drug-disease association edges that belong to the test set from the graph.
    test_id: numpy array shape [n,2] -> [drug_index, disease_index]
    """
    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]

    edges_id = g.edge_ids(
        th.tensor(test_drug_id),
        th.tensor(test_dis_id),
        etype=("drug", "drug_disease", "disease"),
    )
    g = dgl.remove_edges(g, edges_id, etype=("drug", "drug_disease", "disease"))

    edges_id = g.edge_ids(
        th.tensor(test_dis_id),
        th.tensor(test_drug_id),
        etype=("disease", "disease_drug", "drug"),
    )
    g = dgl.remove_edges(g, edges_id, etype=("disease", "disease_drug", "drug"))
    return g
