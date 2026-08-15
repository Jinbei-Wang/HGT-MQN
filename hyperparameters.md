# Hyperparameter Settings

This file summarizes the final hyperparameter configurations used for the benchmark experiments of HGT-MQN. Parameters not explicitly specified for a dataset follow the default values defined in the corresponding training script.

## Cdataset

| Parameter | Value |
|---|---:|
| `bce_pos_weight_scale` | 1.5 |
| `cl_sample_size` | 0 |
| `cl_temperature` | 0.2 |
| `dropout` | 0.2 |
| `epoch` | 500 |
| `feature_mode` | `llm` |
| `grad_clip` | 1.0 |
| `hidden_feats` | 128 |
| `lambda_cl_het` | 0.1 |
| `lambda_cl_sim` | 0.0 |
| `layer_pooling` | `attn` |
| `learning_rate` | 5e-4 |
| `neg_ratio` | 30 |
| `nfold` | 10 |
| `num_heads` | 4 |
| `num_hgt_layers` | 3 |
| `num_sim_layers` | 2 |
| `pair_hidden` | 128 |
| `pair_mode` | `rotate` |
| `patience` | 30 |
| `query_gamma_init` | 0.05 |
| `query_layers` | `1` |
| `seed` | 42 |
| `sim_diffusion_alpha` | 0.15 |
| `sim_diffusion_steps` | 3 |
| `sim_init_mode` | `sim_feature` |
| `sim_learning_rate` | 1e-3 |
| `sim_no_diffusion` | `False` |
| `sim_topk` | 5 |
| `sim_use_diffused_adj_for_gcn` | `False` |
| `sim_use_diffusion` | `False` |
| `weight_decay` | 1e-5 |

## Kdataset

| Parameter | Value |
|---|---:|
| `bce_pos_weight_scale` | 1.5 |
| `cl_sample_size` | 0 |
| `cl_temperature` | 0.2 |
| `dropout` | 0.2 |
| `epoch` | 500 |
| `feature_mode` | `llm` |
| `grad_clip` | 1.0 |
| `hidden_feats` | 128 |
| `lambda_cl_het` | 0.1 |
| `lambda_cl_sim` | 0.0 |
| `layer_pooling` | `attn` |
| `learning_rate` | 5e-4 |
| `neg_ratio` | 30 |
| `nfold` | 10 |
| `num_heads` | 4 |
| `num_hgt_layers` | 3 |
| `num_sim_layers` | 2 |
| `pair_hidden` | 128 |
| `pair_mode` | `rotate` |
| `patience` | 30 |
| `query_gamma_init` | 0.05 |
| `query_layers` | `1` |
| `seed` | 42 |
| `sim_diffusion_alpha` | 0.15 |
| `sim_diffusion_steps` | 3 |
| `sim_init_mode` | `sim_feature` |
| `sim_learning_rate` | 1e-3 |
| `sim_no_diffusion` | `False` |
| `sim_topk` | 5 |
| `sim_use_diffused_adj_for_gcn` | `False` |
| `sim_use_diffusion` | `False` |
| `weight_decay` | 1e-5 |

## Bdataset

The following values were explicitly specified in the final Bdataset training command.

| Parameter | Value |
|---|---:|
| `bce_pos_weight_scale` | 1.0 |
| `cl_temperature` | 0.2 |
| `epoch` | 300 |
| `grad_clip` | 1.0 |
| `hidden_feats` | 128 |
| `lambda_cl_het` | 0.0 |
| `lambda_cl_sim` | 0.0 |
| `layer_pooling` | `attn` |
| `learning_rate` | 1e-4 |
| `neg_ratio` | 6 |
| `num_hgt_layers` | 3 |
| `num_sim_layers` | 2 |
| `query_gamma_init` | 0.05 |
| `query_layers` | `1` |
| `sim_learning_rate` | 1e-3 |
| `sim_topk` | 5 |
| `weight_decay` | 1e-5 |

Parameters not listed above use the defaults defined in the training script.

## Notes

- Conventional benchmark experiments use 10-fold cross-validation where specified.
- The random seed for Cdataset and Kdataset is fixed to `42`.
- `sim_topk = 5` is used to construct the similarity neighborhood.
- `query_layers = 1` indicates the selected HGT layer for memory-query injection according to the implementation.
- Cold-start-specific parameters are configured separately in `cold_start.py`.
- The hyperparameter search space used in the manuscript is reported in Supplementary Table S1.
