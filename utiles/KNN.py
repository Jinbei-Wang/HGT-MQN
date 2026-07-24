import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

#TOPK没有把自己包含
@torch.no_grad()
def knn_from_baseline_sim(baseline_sim: torch.Tensor, K: int = 15) -> torch.Tensor:
    """baseline_sim: [N,N] -> knn_sim: [N,K]"""
    assert baseline_sim.dim() == 2 and baseline_sim.size(0) == baseline_sim.size(1)
    N = baseline_sim.size(0)
    K = min(K, N - 1)

    S = baseline_sim.clone()
    S.fill_diagonal_(-float("inf"))
    _, idx = torch.topk(S, k=K, dim=1, largest=True, sorted=True)
    return idx.long()


@torch.no_grad()
def knn_from_llm_emb(llm_emb: torch.Tensor, K: int = 15) -> torch.Tensor:
    """llm_emb: [N,d] -> knn_het: [N,K]（余弦TopK）"""
    assert llm_emb.dim() == 2
    N = llm_emb.size(0)
    K = min(K, N - 1)

    X = F.normalize(llm_emb, p=2, dim=1)
    S = X @ X.t()
    S.fill_diagonal_(-float("inf"))
    _, idx = torch.topk(S, k=K, dim=1, largest=True, sorted=True)
    return idx.long()


def load_baseline_sim_csv(path: str, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """读取 baseline 相似度矩阵 CSV -> [N,N] tensor"""
    df = pd.read_csv(path, header=None).fillna(0.0)
    S = torch.tensor(df.to_numpy(), dtype=dtype, device=device)
    return S


def load_llm_emb_npy(path: str, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """读取对齐后的 LLM embedding npy -> [N,d] tensor"""
    X = np.load(path)
    return torch.from_numpy(X).to(device=device, dtype=dtype)


@torch.no_grad()
def prepare_knn_topk(
    *,
    baseline_csv_path: str,
    llm_emb_npy_path: str,
    K: int = 15,
    device: torch.device = torch.device("cuda"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    一步到位：
    - 读取 baseline CSV -> baseline_sim
    - 读取 llm_emb npy
    - 计算 knn_sim / knn_het
    返回：
      knn_sim: [N,K] long
      knn_het: [N,K] long
    """
    baseline_sim = load_baseline_sim_csv(baseline_csv_path, device=device)
    llm_emb = load_llm_emb_npy(llm_emb_npy_path, device=device)

    knn_sim = knn_from_baseline_sim(baseline_sim, K=K)
    knn_het = knn_from_llm_emb(llm_emb, K=K)
    return knn_sim, knn_het

# device = torch.device("cuda")  # 默认GPU
#
# base = r"dataset\Kdataset"
# knn_sim_drug, knn_het_drug = prepare_knn_topk(
#     baseline_csv_path=os.path.join(base, "drug_drug_baseline.csv"),
#     llm_emb_npy_path=os.path.join(base, "drug_LLM_emb_aligned.npy"),
#     K=15,
#     device=device,
# )
#
# knn_sim_dis, knn_het_dis = prepare_knn_topk(
#     baseline_csv_path=os.path.join(base, "disease_disease_baseline.csv"),
#     llm_emb_npy_path=os.path.join(base, "disease_LLM_emb_aligned.npy"),
#     K=15,
#     device=device,
# )