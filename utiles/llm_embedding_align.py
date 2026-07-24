# align_llm_emb.py
import os
import numpy as np
import pandas as pd

def align_pkl_to_csv(pkl_path: str, csv_path: str, id_col: str, out_npy_path: str):
    emb = pd.read_pickle(pkl_path)
    if not isinstance(emb, dict):
        raise TypeError(f"{pkl_path} is not dict, got {type(emb)}")
    emb = {str(k): np.asarray(v, dtype=np.float32) for k, v in emb.items()}
    dim = next(iter(emb.values())).shape[0]

    df = pd.read_csv(csv_path)
    ids = df[id_col].astype(str).tolist()

    X = np.zeros((len(ids), dim), dtype=np.float32)
    missing = []
    for i, node_id in enumerate(ids):
        v = emb.get(node_id)
        if v is None:
            missing.append(node_id)
        else:
            X[i] = v

    print(f"[ALIGN] {os.path.basename(pkl_path)} -> {X.shape} | missing={len(missing)}")
    if missing:
        print("  missing examples:", missing[:10])

    np.save(out_npy_path, X)
    print("[SAVE]", out_npy_path)
    return X

if __name__ == "__main__":
    base = r"dataset\Kdataset"

    align_pkl_to_csv(
        pkl_path=os.path.join(base, "LLM_drug_emb.pkl"),
        csv_path=os.path.join(base, "omics", "drug.csv"),   # 你自己的文件名
        id_col="Drug",
        out_npy_path=os.path.join(base, "drug_LLM_emb_aligned.npy"),
    )

    align_pkl_to_csv(
        pkl_path=os.path.join(base, "LLM_disease_emb.pkl"),
        csv_path=os.path.join(base, "omics", "disease.csv"),
        id_col="Disease",
        out_npy_path=os.path.join(base, "disease_LLM_emb_aligned.npy"),
    )
