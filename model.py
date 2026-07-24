import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


# ============================================================
# Utilities
# ============================================================

def build_homo_pack_from_inputs(graph, inputs: Dict[str, torch.Tensor]):
    """Build a reusable homogeneous-graph pack for DGL HGTConv."""
    homo_graph = dgl.to_homogeneous(graph)
    node_ranges = {}
    start = 0
    for ntype in graph.ntypes:
        n = graph.num_nodes(ntype)
        node_ranges[ntype] = slice(start, start + n)
        start += n
    return {
        "g_homo": homo_graph,
        "ntype_ids": homo_graph.ndata[dgl.NTYPE],
        "etype_ids": homo_graph.edata[dgl.ETYPE],
        "node_ranges": node_ranges,
        "ntypes": list(graph.ntypes),
    }


def row_normalize(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return mat / mat.sum(dim=1, keepdim=True).clamp_min(eps)


def build_topk_from_similarity(S: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S: [N, N] similarity matrix. Diagonal is excluded from top-k memory.
    Returns topk_idx/topk_weight, each node has K similar-neighbor memory tokens.
    """
    N = S.size(0)
    k = max(1, min(int(k), N - 1))
    S_work = S.clone()
    S_work.fill_diagonal_(-float("inf"))
    topk_scores, topk_idx = torch.topk(S_work, k=k, dim=1, largest=True, sorted=True)
    # If scores are normalized similarities, softmax gives stable memory weights.
    topk_weight = torch.softmax(topk_scores, dim=1)
    return topk_idx, topk_weight


def build_topk_adjacency(S: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
    """Dense top-k adjacency used by SimGCN layers."""
    N = S.size(0)
    A = torch.zeros_like(S)
    row_idx = torch.arange(N, device=S.device).unsqueeze(1).expand_as(topk_idx)
    A[row_idx, topk_idx] = topk_weight
    A = A + torch.eye(N, device=S.device, dtype=S.dtype)
    return row_normalize(A)


def ppr_diffuse(S: torch.Tensor, alpha: float = 0.15, steps: int = 5) -> torch.Tensor:
    """PPR-like diffusion over a normalized similarity matrix."""
    if steps <= 0:
        return S
    N = S.size(0)
    I = torch.eye(N, device=S.device, dtype=S.dtype)
    Z = I
    for _ in range(steps):
        Z = alpha * I + (1.0 - alpha) * (S @ Z)
    return row_normalize(Z)


def gather_topk_tokens(all_tokens: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    """
    all_tokens: [N, D]
    topk_idx: [N, K]
    return: [N, K, D]
    """
    return all_tokens[topk_idx]


# ============================================================
# HGT global encoder blocks
# ============================================================

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

    def forward(self, inputs: Dict[str, torch.Tensor], homo_pack):
        feats = torch.cat([inputs[ntype] for ntype in homo_pack["ntypes"]], dim=0)
        device = feats.device
        g_homo = homo_pack["g_homo"]
        ntype_ids = homo_pack["ntype_ids"]
        etype_ids = homo_pack["etype_ids"]
        if g_homo.device != device:
            g_homo = g_homo.to(device)
            ntype_ids = ntype_ids.to(device)
            etype_ids = etype_ids.to(device)
        h_all = self.hgt(g_homo, feats, ntype_ids, etype_ids, presorted=True)
        out = {}
        for ntype, slc in homo_pack["node_ranges"].items():
            x = h_all[slc]
            x = self.bn[ntype](x)
            x = self.dropout(x)
            out[ntype] = self.act(x)
        return out

# ============================================================
# Similarity Graph Transformer encoder
# ============================================================

class SimGraphTransformerLayer(nn.Module):
    """
    Dense similarity Graph Transformer layer.

    输入:
        x:   [N, D]
        adj: [N, N] top-k normalized similarity adjacency

    这里不用 DGL message passing，而是直接在 dense top-k adjacency mask 上做 attention。
    好处:
      1. 不容易出现 DGL u_mul_e 维度广播错误；
      2. 和原来的 SimGCN 一样使用 dense A_d / A_s；
      3. 可以直接复用原来的 top-k similarity adjacency；
      4. 输出仍然是 [N, D]，可以直接作为 QueryPoolingBlock 的 memory tokens。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_bias: bool = True,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [N, D]
        adj: [N, N], row-normalized top-k similarity adjacency.
             adj[i, j] > 0 表示节点 i 可以 attend 到节点 j。
        """
        N, D = x.shape

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # q/k/v: [H, N, Dh]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # attn_logits: [H, N, N]

        # 只允许 top-k similarity 邻居参与 attention
        mask = adj > 0
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(0), -1e9)

        # 用 similarity score 作为 edge bias
        if self.use_edge_bias:
            edge_bias = torch.log(adj.clamp_min(1e-8)).unsqueeze(0)
            attn_logits = attn_logits + edge_bias

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        # out: [H, N, Dh]

        out = out.transpose(0, 1).contiguous().view(N, D)
        out = self.o_proj(out)

        x = self.norm1(x + self.out_drop(out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.out_drop(ffn_out))

        return x


class SimilarityGraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        topk: int = 10,
        use_diffusion: bool = True,
        diffusion_alpha: float = 0.15,
        diffusion_steps: int = 3,
        use_diffused_adj_for_gcn: bool = False,
        use_edge_bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.topk = int(topk)

        self.use_diffusion = use_diffusion
        self.diffusion_alpha = diffusion_alpha
        self.diffusion_steps = diffusion_steps
        self.use_diffused_adj_for_gcn = use_diffused_adj_for_gcn
        self.use_edge_bias = use_edge_bias

        self.drug_layers = nn.ModuleList([
            SimGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_bias=use_edge_bias,
            )
            for _ in range(self.num_layers)
        ])

        self.disease_layers = nn.ModuleList([
            SimGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_bias=use_edge_bias,
            )
            for _ in range(self.num_layers)
        ])

        self.register_buffer("S_d", torch.empty(0), persistent=False)
        self.register_buffer("S_s", torch.empty(0), persistent=False)
        self.register_buffer("A_d", torch.empty(0), persistent=False)
        self.register_buffer("A_s", torch.empty(0), persistent=False)

        self.register_buffer("drug_topk_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("disease_topk_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("drug_topk_weight", torch.empty(0), persistent=False)
        self.register_buffer("disease_topk_weight", torch.empty(0), persistent=False)

    def set_similarity_from_graphs(
        self,
        drdr_graph,
        didi_graph,
        device: Optional[torch.device] = None,
    ):
        if "sim_feature" not in drdr_graph.ndata:
            raise KeyError("drdr_graph.ndata['sim_feature'] not found")
        if "sim_feature" not in didi_graph.ndata:
            raise KeyError("didi_graph.ndata['sim_feature'] not found")

        device = device or next(self.parameters()).device

        S_d = drdr_graph.ndata["sim_feature"].float().to(device)
        S_s = didi_graph.ndata["sim_feature"].float().to(device)

        S_d = S_d + torch.eye(S_d.size(0), device=device, dtype=S_d.dtype)
        S_s = S_s + torch.eye(S_s.size(0), device=device, dtype=S_s.dtype)

        S_d = row_normalize(S_d)
        S_s = row_normalize(S_s)

        S_d_mem = (
            ppr_diffuse(S_d, self.diffusion_alpha, self.diffusion_steps)
            if self.use_diffusion
            else S_d
        )
        S_s_mem = (
            ppr_diffuse(S_s, self.diffusion_alpha, self.diffusion_steps)
            if self.use_diffusion
            else S_s
        )

        drug_topk_idx, drug_topk_weight = build_topk_from_similarity(
            S_d_mem,
            self.topk,
        )
        disease_topk_idx, disease_topk_weight = build_topk_from_similarity(
            S_s_mem,
            self.topk,
        )

        # 这里沿用原参数名 use_diffused_adj_for_gcn，
        # 但现在含义是：Graph Transformer attention mask/bias 是否使用 diffused similarity。
        A_source_d = S_d_mem if self.use_diffused_adj_for_gcn else S_d
        A_source_s = S_s_mem if self.use_diffused_adj_for_gcn else S_s

        A_d = build_topk_adjacency(
            A_source_d,
            drug_topk_idx,
            drug_topk_weight,
        )
        A_s = build_topk_adjacency(
            A_source_s,
            disease_topk_idx,
            disease_topk_weight,
        )

        self.S_d = S_d_mem
        self.S_s = S_s_mem
        self.A_d = A_d
        self.A_s = A_s

        self.drug_topk_idx = drug_topk_idx
        self.disease_topk_idx = disease_topk_idx
        self.drug_topk_weight = drug_topk_weight
        self.disease_topk_weight = disease_topk_weight

    def _ensure_ready(self):
        if self.A_d.numel() == 0 or self.A_s.numel() == 0:
            raise RuntimeError(
                "Similarity matrices have not been set. "
                "Call model.set_similarity_graphs(...)."
            )

    def forward(
        self,
        drug_x: torch.Tensor,
        disease_x: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:

        self._ensure_ready()

        drug_layers = []
        disease_layers = []

        hd = drug_x
        hs = disease_x

        for l in range(self.num_layers):
            hd = self.drug_layers[l](hd, self.A_d)
            hs = self.disease_layers[l](hs, self.A_s)

            drug_layers.append(hd)
            disease_layers.append(hs)

        return {
            "drug_layers": drug_layers,
            "disease_layers": disease_layers,

            "drug_topk_idx": self.drug_topk_idx,
            "disease_topk_idx": self.disease_topk_idx,

            "drug_topk_weight": self.drug_topk_weight,
            "disease_topk_weight": self.disease_topk_weight,
        }

# ============================================================
# Query pooling from similarity memory
# ============================================================

class QueryPoolingBlock(nn.Module):
    """
    HGT global node features query top-k sim-GCN memory tokens.
    Only drug/disease nodes are updated.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, gamma_init: float = 0.05):
        super().__init__()
        self.drug_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.disease_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.drug_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.disease_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.drug_norm = nn.LayerNorm(hidden_dim)
        self.disease_norm = nn.LayerNorm(hidden_dim)
        gamma_init_t = torch.tensor(float(gamma_init)).clamp(1e-4, 1.0 - 1e-4)
        self.gamma_raw = nn.Parameter(torch.logit(gamma_init_t))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)

    def _inject_one(self, q_feat, memory_tokens, memory_weight, attn, gate_net, norm):
        # q_feat: [N,D], memory_tokens: [N,K,D], memory_weight: [N,K]
        q = q_feat.unsqueeze(1)
        kv = memory_tokens * memory_weight.unsqueeze(-1)
        ctx, attn_w = attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        ctx = ctx.squeeze(1)
        gate_input = torch.cat([q_feat, ctx, q_feat * ctx, torch.abs(q_feat - ctx)], dim=-1)
        gate = gate_net(gate_input)
        out = norm(q_feat + self.gamma * gate * ctx)
        return out, ctx, gate, attn_w

    def forward(
        self,
        global_nodes: Dict[str, torch.Tensor],
        sim_layer: Dict[str, torch.Tensor],
        topk_pack: Dict[str, torch.Tensor],
        return_aux: bool = False,
    ):
        out = {k: v for k, v in global_nodes.items()}
        aux = {}
        drug_tokens = gather_topk_tokens(sim_layer["drug"], topk_pack["drug_topk_idx"])
        disease_tokens = gather_topk_tokens(sim_layer["disease"], topk_pack["disease_topk_idx"])

        out_drug, drug_ctx, drug_gate, drug_attn = self._inject_one(
            global_nodes["drug"],
            drug_tokens,
            topk_pack["drug_topk_weight"],
            self.drug_attn,
            self.drug_gate,
            self.drug_norm,
        )
        out_dis, disease_ctx, disease_gate, disease_attn = self._inject_one(
            global_nodes["disease"],
            disease_tokens,
            topk_pack["disease_topk_weight"],
            self.disease_attn,
            self.disease_gate,
            self.disease_norm,
        )
        out["drug"] = out_drug
        out["disease"] = out_dis
        if return_aux:
            aux.update({
                "drug_ctx": drug_ctx,
                "disease_ctx": disease_ctx,
                "drug_gate_mean": drug_gate.detach().mean(),
                "disease_gate_mean": disease_gate.detach().mean(),
                "drug_attn": drug_attn,
                "disease_attn": disease_attn,
                "gamma": self.gamma.detach(),
            })
            return out, aux
        return out


# ============================================================
# Pooling, decoder, contrastive loss
# ============================================================
class LayerAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: int = 32):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )

    def forward(self, layers: List[torch.Tensor], return_attention: bool = False):
        z = torch.stack(layers, dim=1)  # [N,L,D]
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        out = (beta * z).sum(dim=1)
        if return_attention:
            return out, beta
        return out


def pool_layers(layers: List[torch.Tensor], mode: str, attn_pool: Optional[LayerAttentionPooling] = None):
    if len(layers) == 0:
        raise ValueError("No layer outputs to pool.")
    if mode == "last":
        return layers[-1]
    if mode == "mean":
        return torch.stack(layers, dim=0).mean(dim=0)
    if mode == "dream":
        out = layers[0]
        for i in range(1, len(layers)):
            out = out + layers[i] / float(i + 1)
        return out
    if mode == "attn":
        if attn_pool is None:
            raise ValueError("attn_pool is required when mode='attn'.")
        return attn_pool(layers)
    raise ValueError(f"Unsupported layer pooling mode: {mode}")


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
        return torch.cat([a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re], dim=-1)

    def pair_feature(self, drug_h, disease_h):
        mul = drug_h * disease_h
        last = self.rotate_operator(drug_h, disease_h) if self.pair_mode == "rotate" else torch.abs(drug_h - disease_h)
        return torch.cat([drug_h, disease_h, mul, last], dim=-1)

    def forward(self, h_nodes: Dict[str, torch.Tensor], drug_idx: torch.Tensor, disease_idx: torch.Tensor):
        pair_feat = self.pair_feature(h_nodes["drug"][drug_idx], h_nodes["disease"][disease_idx])
        logit = self.mlp(pair_feat).squeeze(-1)
        return logit, pair_feat


def masked_symmetric_info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    topk_idx: torch.Tensor,
    temperature: float = 0.2,
):
    """
    anchor:   [N, D] query-enhanced global features
    positive: [N, D] raw HGT features or sim-GCN features
    topk_idx: [N, K] top-k similar neighbor indices for each node

    Positive:
        anchor_i <-> positive_i

    Negatives:
        all nodes except i and top-k similar neighbors of i
    """
    n = anchor.size(0)
    device = anchor.device

    anchor = torch.nn.functional.normalize(anchor, dim=-1)
    positive = torch.nn.functional.normalize(positive, dim=-1)

    labels = torch.arange(n, device=device)

    # ------------------------------------------------------------
    # anchor -> positive
    # ------------------------------------------------------------
    logits_ap = anchor @ positive.t()
    logits_ap = logits_ap / temperature

    # mask top-k similar nodes as "ignored negatives"
    mask_ap = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask_ap.scatter_(1, topk_idx.to(device), True)

    # keep diagonal as positive, do not mask self
    mask_ap[labels, labels] = False

    logits_ap = logits_ap.masked_fill(mask_ap, -1e9)
    loss_ap = torch.nn.functional.cross_entropy(logits_ap, labels)

    # ------------------------------------------------------------
    # positive -> anchor
    # ------------------------------------------------------------
    logits_pa = positive @ anchor.t()
    logits_pa = logits_pa / temperature

    mask_pa = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask_pa.scatter_(1, topk_idx.to(device), True)
    mask_pa[labels, labels] = False

    logits_pa = logits_pa.masked_fill(mask_pa, -1e9)
    loss_pa = torch.nn.functional.cross_entropy(logits_pa, labels)

    return 0.5 * (loss_ap + loss_pa)


# ============================================================
# Main model
# ============================================================

class HGTSimGTQueryCLModel(nn.Module):
    """
    Main clean model:
      - HGT global branch, L layers.
      - Similarity GT branch, L layers, on drug-drug and disease-disease similarity graphs.
      - By default, SimGT uses dr_graph/di_graph ndata["sim_feature"] as its own initial features,
        instead of reusing HGT LLM-projected drug/disease features.
      - After each HGT layer, drug/disease global nodes query top-k sim memory from the corresponding sim-GCN layer.
      - Final prediction uses only query-enhanced global HGT representations, with optional layer pooling.
      - Optional InfoNCE losses are computed outside using aux features returned by forward.
    """

    @staticmethod
    def _parse_query_layers(query_layers: str, num_hgt_layers: int):
        """
        query_layers:
            "all"  -> all HGT layers use query pooling
            "none" -> no query pooling
            "0,1"  -> only HGT layer 0 and 1 use query pooling
        """
        if query_layers is None:
            return set(range(num_hgt_layers))
        query_layers = str(query_layers).strip().lower()
        if query_layers == "all":
            return set(range(num_hgt_layers))
        if query_layers == "none":
            return set()
        layer_ids = set()
        for x in query_layers.split(","):
            x = x.strip()
            if x == "":
                continue
            idx = int(x)
            if idx < 0 or idx >= num_hgt_layers:
                raise ValueError(
                    f"Invalid query layer {idx}. "
                    f"Valid range is [0, {num_hgt_layers - 1}]."
                )
            layer_ids.add(idx)
        return layer_ids

    def __init__(
        self,
        etypes,
        ntypes,
        n_drug: int,
        n_dis: int,
        in_feats: int,
        hidden_feats: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_hgt_layers: int = 2,
        num_sim_layers: int = 2,
        query_layers: str = "all",
        pair_hidden: int = 128,
        pair_mode: str = "rotate",
        drug_in_dim: int = 1536,
        disease_in_dim: int = 1536,
        sim_drug_in_dim: Optional[int] = None,
        sim_disease_in_dim: Optional[int] = None,
        sim_init_mode: str = "sim_feature",
        sim_topk: int = 10,
        sim_use_diffusion: bool = True,
        sim_diffusion_alpha: float = 0.15,
        sim_diffusion_steps: int = 3,
        sim_use_diffused_adj_for_gcn: bool = False,
        query_gamma_init: float = 0.05,
        layer_pooling: str = "dream",
        cl_temperature: float = 0.2,
        cl_sample_size: int = 0,
    ):
        super().__init__()
        if layer_pooling not in ["last", "mean", "dream", "attn"]:
            raise ValueError(f"Unsupported layer_pooling: {layer_pooling}")
        self.ntypes = list(ntypes)
        self.hidden_feats = hidden_feats
        self.num_hgt_layers = int(num_hgt_layers)
        self.num_sim_layers = int(num_sim_layers)
        self.query_layers = self._parse_query_layers(query_layers, self.num_hgt_layers)
        self.layer_pooling = layer_pooling
        self.n_drug = n_drug
        self.n_dis = n_dis
        self.sim_init_mode = str(sim_init_mode).lower()
        if self.sim_init_mode not in ["sim_feature", "shared"]:
            raise ValueError(f"Unsupported sim_init_mode: {sim_init_mode}. Use 'sim_feature' or 'shared'.")

        # HGT-view initialization: drug/disease LLM embeddings from x_het.
        self.drug_linear = nn.Linear(drug_in_dim, hidden_feats)
        self.disease_linear = nn.Linear(disease_in_dim, hidden_feats)
        self.other_linear = nn.Linear(in_feats, hidden_feats)

        # SimGT-view initialization: by default, use similarity profiles from
        # dr_graph.ndata['sim_feature'] and di_graph.ndata['sim_feature'].
        # For K/B/C datasets these dimensions are n_drug and n_dis.
        sim_drug_in_dim = int(sim_drug_in_dim) if sim_drug_in_dim is not None else int(n_drug)
        sim_disease_in_dim = int(sim_disease_in_dim) if sim_disease_in_dim is not None else int(n_dis)
        self.sim_drug_feature_linear = nn.Linear(sim_drug_in_dim, hidden_feats)
        self.sim_disease_feature_linear = nn.Linear(sim_disease_in_dim, hidden_feats)

        for layer in [
            self.drug_linear,
            self.disease_linear,
            self.other_linear,
            self.sim_drug_feature_linear,
            self.sim_disease_feature_linear,
        ]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.hgt_layers = nn.ModuleList([
            HGTLayer(hidden_feats, ntypes, etypes, num_heads=num_heads, dropout=dropout)
            for _ in range(self.num_hgt_layers)
        ])

        self.sim_encoder = SimilarityGraphTransformerEncoder(
            hidden_dim=hidden_feats,
            num_layers=self.num_sim_layers,
            num_heads=num_heads,
            dropout=dropout,
            topk=sim_topk,
            use_diffusion=sim_use_diffusion,
            diffusion_alpha=sim_diffusion_alpha,
            diffusion_steps=sim_diffusion_steps,
            use_diffused_adj_for_gcn=sim_use_diffused_adj_for_gcn,
            use_edge_bias=True,
        )

        self.query_blocks = nn.ModuleList([
            QueryPoolingBlock(hidden_feats, num_heads=num_heads, dropout=dropout, gamma_init=query_gamma_init)
            for _ in range(self.num_hgt_layers)
        ])

        self.drug_layer_attn = LayerAttentionPooling(hidden_feats)
        self.disease_layer_attn = LayerAttentionPooling(hidden_feats)
        self.sim_drug_layer_attn = LayerAttentionPooling(hidden_feats)
        self.sim_disease_layer_attn = LayerAttentionPooling(hidden_feats)

        self.pair_decoder = PairDecoder(
            hidden_feats,
            pair_hidden=pair_hidden,
            dropout=dropout,
            pair_mode=pair_mode,
        )

        self.cl_temperature = float(cl_temperature)
        self.cl_sample_size = int(cl_sample_size)

    def set_similarity_graphs(self, drdr_graph, didi_graph, device: Optional[torch.device] = None):
        self.sim_encoder.set_similarity_from_graphs(drdr_graph, didi_graph, device=device)

    def _project_inputs(self, x_het: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = {}
        for ntype in self.ntypes:
            if ntype == "drug":
                h[ntype] = self.drug_linear(x_het[ntype])
            elif ntype == "disease":
                h[ntype] = self.disease_linear(x_het[ntype])
            else:
                h[ntype] = self.other_linear(x_het[ntype])
        return h

    def _project_similarity_inputs(
        self,
        h0: Dict[str, torch.Tensor],
        drdr_graph,
        didi_graph,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project SimGT initial features.

        sim_init_mode='sim_feature':
            use drdr_graph.ndata['sim_feature'] / didi_graph.ndata['sim_feature']
            as similarity-profile initial vectors.

        sim_init_mode='shared':
            use the HGT-view projected drug/disease features h0['drug'] / h0['disease'];
            this is kept only for ablation/backward compatibility.
        """
        if self.sim_init_mode == "shared":
            return h0["drug"], h0["disease"]

        if drdr_graph is None or didi_graph is None:
            raise RuntimeError("drdr_graph/didi_graph are required when sim_init_mode='sim_feature'.")
        if "sim_feature" not in drdr_graph.ndata:
            raise KeyError("drdr_graph.ndata['sim_feature'] not found. Check prepare_similarity_graphs().")
        if "sim_feature" not in didi_graph.ndata:
            raise KeyError("didi_graph.ndata['sim_feature'] not found. Check prepare_similarity_graphs().")

        drug_sim_feat = drdr_graph.ndata["sim_feature"].float().to(h0["drug"].device)
        disease_sim_feat = didi_graph.ndata["sim_feature"].float().to(h0["disease"].device)

        if drug_sim_feat.size(1) != self.sim_drug_feature_linear.in_features:
            raise ValueError(
                f"Drug sim_feature dim mismatch: got {drug_sim_feat.size(1)}, "
                f"expected {self.sim_drug_feature_linear.in_features}."
            )
        if disease_sim_feat.size(1) != self.sim_disease_feature_linear.in_features:
            raise ValueError(
                f"Disease sim_feature dim mismatch: got {disease_sim_feat.size(1)}, "
                f"expected {self.sim_disease_feature_linear.in_features}."
            )

        return self.sim_drug_feature_linear(drug_sim_feat), self.sim_disease_feature_linear(disease_sim_feat)

    def _pool_drug_disease(self, drug_layers: List[torch.Tensor], disease_layers: List[torch.Tensor], mode: str):
        drug_pool = pool_layers(drug_layers, mode, self.drug_layer_attn if mode == "attn" else None)
        disease_pool = pool_layers(disease_layers, mode, self.disease_layer_attn if mode == "attn" else None)
        return {"drug": drug_pool, "disease": disease_pool}

    def _pool_sim(self, sim_drug_layers: List[torch.Tensor], sim_disease_layers: List[torch.Tensor], mode: str):
        drug_pool = pool_layers(sim_drug_layers, mode, self.sim_drug_layer_attn if mode == "attn" else None)
        disease_pool = pool_layers(sim_disease_layers, mode, self.sim_disease_layer_attn if mode == "attn" else None)
        return {"drug": drug_pool, "disease": disease_pool}

    def contrastive_loss(
            self,
            aux: Dict[str, object],
            lambda_cl_het: float = 0.0,
            lambda_cl_sim: float = 0.0,
    ):
        """
        Two separated CL losses:
          1) query-enhanced global vs raw HGT global: lambda_cl_het
          2) query-enhanced global vs sim GCN: lambda_cl_sim

        Masked InfoNCE:
          positive = same node across views
          ignored negatives = top-k similar neighbors
          valid negatives = nodes outside top-k
        """
        device = aux["final_node_emb"]["drug"].device
        zero = torch.tensor(0.0, device=device)

        cl_het = zero
        cl_sim = zero

        final_nodes = aux["final_node_emb"]
        raw_nodes = aux["raw_global_pooled"]
        sim_nodes = aux["sim_pooled"]

        drug_topk_idx = aux["drug_topk_idx"]
        disease_topk_idx = aux["disease_topk_idx"]
        assert isinstance(drug_topk_idx, torch.Tensor)
        assert isinstance(disease_topk_idx, torch.Tensor)

        if lambda_cl_het > 0:
            cl_het = (
                    masked_symmetric_info_nce(
                        final_nodes["drug"],
                        raw_nodes["drug"],
                        drug_topk_idx,
                        temperature=self.cl_temperature,
                    )
                    + masked_symmetric_info_nce(
                final_nodes["disease"],
                raw_nodes["disease"],
                disease_topk_idx,
                temperature=self.cl_temperature,
            )
            )

        if lambda_cl_sim > 0:
            cl_sim = (
                    masked_symmetric_info_nce(
                        final_nodes["drug"],
                        sim_nodes["drug"],
                        drug_topk_idx,
                        temperature=self.cl_temperature,
                    )
                    + masked_symmetric_info_nce(
                final_nodes["disease"],
                sim_nodes["disease"],
                disease_topk_idx,
                temperature=self.cl_temperature,
            )
            )

        total = float(lambda_cl_het) * cl_het + float(lambda_cl_sim) * cl_sim
        return total, cl_het, cl_sim

    @torch.no_grad()
    def forward_hgt_raw(self, g_het, x_het: Dict[str, torch.Tensor], drug_idx: torch.Tensor, disease_idx: torch.Tensor):
        """Diagnostic HGT-only forward using raw HGT pooled features before query pooling."""
        out = self.forward(g_het, x_het, None, None, drug_idx, disease_idx, return_aux=True, skip_similarity=True)
        _, aux = out
        logit, _ = self.pair_decoder(aux["raw_global_pooled"], drug_idx, disease_idx)
        return logit

    def forward(
        self,
        g_het,
        x_het: Dict[str, torch.Tensor],
        drdr_graph,
        didi_graph,
        drug_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        return_aux: bool = False,
        skip_similarity: bool = False,
    ):
        h0 = self._project_inputs(x_het)
        h_global = {k: v for k, v in h0.items()}
        homo_pack = build_homo_pack_from_inputs(g_het, h_global)

        # Sim branch outputs are layer-matched to HGT.
        # IMPORTANT: by default SimGT uses similarity-profile initial features,
        # not the same LLM-projected features used by HGT.
        if skip_similarity:
            sim_drug_layers = [h0["drug"] for _ in range(self.num_sim_layers)]
            sim_disease_layers = [h0["disease"] for _ in range(self.num_sim_layers)]
            topk_pack = None
        else:
            sim_drug_x, sim_disease_x = self._project_similarity_inputs(h0, drdr_graph, didi_graph)
            sim_out = self.sim_encoder(sim_drug_x, sim_disease_x)
            sim_drug_layers = sim_out["drug_layers"]
            sim_disease_layers = sim_out["disease_layers"]
            topk_pack = {
                "drug_topk_idx": sim_out["drug_topk_idx"],
                "disease_topk_idx": sim_out["disease_topk_idx"],
                "drug_topk_weight": sim_out["drug_topk_weight"],
                "disease_topk_weight": sim_out["disease_topk_weight"],
            }

        raw_drug_layers, raw_disease_layers = [], []
        query_drug_layers, query_disease_layers = [], []
        query_aux_list = []

        for l in range(self.num_hgt_layers):
            h_global = self.hgt_layers[l](h_global, homo_pack)

            raw_drug_layers.append(h_global["drug"])
            raw_disease_layers.append(h_global["disease"])

            if (not skip_similarity) and (l in self.query_layers):

                sorted_query_layers = sorted(list(self.query_layers))

                if len(sorted_query_layers) == 1:
                    # 单层注入：默认使用 sim 最后一层
                    sim_l = self.num_sim_layers - 1
                else:
                    # 多层注入：按 query_layers 的顺序依次对应 sim 层
                    query_pos = sorted_query_layers.index(l)
                    sim_l = min(query_pos, self.num_sim_layers - 1)

                sim_layer = {
                    "drug": sim_drug_layers[sim_l],
                    "disease": sim_disease_layers[sim_l],
                }

                if return_aux:
                    h_global, q_aux = self.query_blocks[l](
                        h_global,
                        sim_layer,
                        topk_pack,
                        return_aux=True,
                    )
                    q_aux["hgt_layer"] = torch.tensor(l, device=h_global["drug"].device)
                    q_aux["sim_layer"] = torch.tensor(sim_l, device=h_global["drug"].device)
                    query_aux_list.append(q_aux)
                else:
                    h_global = self.query_blocks[l](
                        h_global,
                        sim_layer,
                        topk_pack,
                        return_aux=False,
                    )

            query_drug_layers.append(h_global["drug"])
            query_disease_layers.append(h_global["disease"])

        final_node_emb = self._pool_drug_disease(query_drug_layers, query_disease_layers, self.layer_pooling)
        raw_global_pooled = self._pool_drug_disease(raw_drug_layers, raw_disease_layers, self.layer_pooling)
        sim_pooled = self._pool_sim(sim_drug_layers, sim_disease_layers, self.layer_pooling)

        logit, pair_feat = self.pair_decoder(final_node_emb, drug_idx, disease_idx)

        if return_aux:
            aux = {
                "final_node_emb": final_node_emb,
                "raw_global_pooled": raw_global_pooled,
                "sim_pooled": sim_pooled,
                "pair_feat": pair_feat,
                "raw_global_layers": {"drug": raw_drug_layers, "disease": raw_disease_layers},
                "query_global_layers": {"drug": query_drug_layers, "disease": query_disease_layers},
                "sim_layers": {"drug": sim_drug_layers, "disease": sim_disease_layers},
                "query_aux": query_aux_list,
                "layer_pooling": self.layer_pooling,
                # ===== for masked InfoNCE =====
                "drug_topk_idx": topk_pack["drug_topk_idx"] if topk_pack is not None else None,
                "disease_topk_idx": topk_pack["disease_topk_idx"] if topk_pack is not None else None,
                "drug_topk_weight": topk_pack["drug_topk_weight"] if topk_pack is not None else None,
                "disease_topk_weight": topk_pack["disease_topk_weight"] if topk_pack is not None else None,

                "num_hgt_layers": self.num_hgt_layers,
                "num_sim_layers": self.num_sim_layers,
                "query_layers": sorted(list(self.query_layers)),
                "sim_init_mode": self.sim_init_mode,
            }
            if len(query_aux_list) > 0:
                gammas = torch.stack([x["gamma"].reshape(()) for x in query_aux_list])
                aux["query_gamma_mean"] = gammas.mean()
                aux["drug_query_gate_mean"] = torch.stack([x["drug_gate_mean"].reshape(()) for x in query_aux_list]).mean()
                aux["disease_query_gate_mean"] = torch.stack([x["disease_gate_mean"].reshape(()) for x in query_aux_list]).mean()
            return logit, aux
        return logit
