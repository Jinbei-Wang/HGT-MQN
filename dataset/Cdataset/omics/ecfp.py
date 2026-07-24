import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for i in fp.GetOnBits():
        arr[i] = 1.0
    return arr


def main():
    csv_path = "drug.csv"   # 改成你的药物文件名
    out_npy = "drug_content_emb.npy"

    df = pd.read_csv(csv_path)
    df = df.sort_values("ID").reset_index(drop=True)

    # 检查 ID 是否从 0 到 N-1 连续
    assert np.array_equal(df["ID"].values, np.arange(len(df))), "ID 列与图节点顺序不连续或不一致"

    feats = []
    for _, row in df.iterrows():
        smiles = row["SMILES"]
        feats.append(smiles_to_ecfp(smiles, radius=2, n_bits=1024))

    feats = np.stack(feats, axis=0).astype(np.float32)
    np.save(out_npy, feats)

    print("Saved:", out_npy, feats.shape)


if __name__ == "__main__":
    main()