# HGT-MQN

HGT-MQN integrates heterogeneous biomedical information and homogeneous similarity information through a memory-guided representation learning framework, enabling drug and disease representations to incorporate complementary biomedical context and transferable similarity patterns.
The framework is shown in the figure below.
<p align="center">
  <img src="framework.png" width="900">
</p>

## Repository Structure.


```text
HGT-MQN/
├── dataset/                       # Benchmark and external evaluation datasets
├── utiles/                        # Utility functions
├── main.py                        # Main training script
├── cold_start.py                  # Cold-start evaluation on benchmark datasets
├── eval_repoapp_to_repoclin.py    # RepoApp-to-RepoClin cross-domain evaluation
├── load_data.py                   # Dataset loading and preprocessing
├── model.py                       # Model architecture
├── train_no_sim.py                # Ablation model without similarity information
├── hyperparameters.md             # Dataset-specific optimal hyperparameters
└── README.md
```

## Environment

The experiments were conducted with the following main dependencies:

- PyTorch 2.4.1+cu121
- CUDA 12.1
- DGL 1.1.2+cu117

## Datasets

All datasets used in this study are organized under the `dataset/` directory.

The benchmark evaluation includes:

* **Cdataset**
* **Kdataset**
* **Bdataset**

The cross-domain cold-start evaluation additionally uses:

* **RepoApp**
* **RepoClin**

Cdataset, Kdataset, and Bdataset are used for conventional and entity-level cold-start evaluation. RepoApp is used as the source dataset for model training, while RepoClin is used as an independent target dataset for cross-domain cold-start evaluation.

## Model Training

The main training procedure is implemented in:

```text
main.py
```

The model is evaluated using **10-fold cross-validation** with a fixed random seed of `42`.

A typical training command follows the form:

```bash
python main.py -da <dataset> -id <gpu_id> -sp <output_directory> [additional arguments]
```

where `<dataset>` can be `Cdataset`, `Kdataset`, `Bdataset`, or `RepoApp`.

## Cold-Start Evaluation

Entity-level cold-start experiments on the three benchmark datasets are implemented in:

```text
cold_start.py
```

The cold-start protocol evaluates the model under both:

* drug cold-start
* disease cold-start

settings using 10-fold entity-level cross-validation.

Test drug or disease entities and their associated interaction edges are excluded from the training graph. Representations for unseen entities are subsequently constructed using transferable information from seen entities.

## Cross-Domain Evaluation

Cross-domain cold-start evaluation from **RepoApp** to **RepoClin** is implemented in:

```text
eval_repoapp_to_repoclin.py
```

The evaluation procedure consists of two stages:

1. Train HGT-MQN on RepoApp using `main.py`.
2. Load the trained RepoApp models and evaluate them on RepoClin using:

```bash
python eval_repoapp_to_repoclin.py
```

This setting evaluates the ability of HGT-MQN to generalize learned representations to an independent drug repositioning dataset.


The complete hyperparameter search space is reported in **Supplementary Table S1** of the manuscript.

## Reproducibility

For reproducibility:

* all benchmark experiments use 10-fold cross-validation;
* the random seed is fixed to `42`;
* dataset-specific optimal hyperparameters are documented in `hyperparameters.md`;
* model checkpoints and experimental outputs are stored in the output directory specified by `-sp`;
* cold-start evaluation follows an entity-level split to prevent test-entity interaction information from entering the training graph.


## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{HGT-MQN,
  title   = {A memory-guided representation learning framework for cold-start drug repositioning towards translational applications},
  author  = {Wang, Jinbei and others},
  journal = {To be updated},
  year    = {2026}
}
```

The citation information will be updated after publication.

## Availability

The source code, processed datasets, model implementation, and experimental configurations associated with this study are provided in this repository.

## Contact

For questions regarding the implementation or experiments, please open an issue in this repository.
