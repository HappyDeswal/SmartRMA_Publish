# Measured results for journal use

Eight figures are supplied in 300-dpi PNG, vector PDF and SVG. Use CAPTIONS.md for publication captions. All are generated from the new project's retained experiment, not copied numerical results from the old project's paper.

## Reproduce

From the project root:

```sh
python -m pip install -r figures/requirements.txt
python figures/generate_figures.py
```

The script checks that all 145 retained prediction records reproduce the recorded confusion matrix and accuracy before plotting. `figure_manifest.json` records SHA-256 provenance. `derived_curve_metrics.json` contains empirical one-vs-rest ROC AUC and average precision computed from saved class probabilities. ROC/PR metrics measure product-type ranking, not physical condition. The plotting code groups tied scores and uses a step PR curve.

The architecture diagram describes implementation rather than benchmark performance. The other seven figures describe one MobileNetV3 Small experiment: 679 training, 147 validation and 145 test images. The selected checkpoint is epoch 9 by validation macro F1. Test accuracy is 122/145 (84.14%), versus 109/145 (75.17%) for a frozen ImageNet centroid baseline. This is not a comparison against Qwen3-VL or the old project. GPU recall declined from 82.61% to 52.17%; SSD 19/19 is a small, source-limited sample.

## Scope and limitations

- No verified damage, authenticity, electrical-function or warranty ground-truth labels were used. These figures establish no damage or end-to-end RMA decision accuracy.
- Exact-pixel duplicate removal and pHash grouping reduce obvious duplicates, but physical-item IDs are unknown. Different views of one unit may cross splits. Background/source shortcuts and unverified source category labels remain possible.
- No repeated-seed study, independent source/item-disjoint evaluation, uncertainty interval or validated out-of-distribution detection is claimed.
- These exploratory findings must not be merged with the old journal's 250-image defect experiment.
- No new policy-RAG benchmark, SLA comparison, localization-error plot or security-ablation radar is supplied: equivalent measured evidence is absent. Software fixture pass counts are not model accuracy.
- Public dataset attribution and licensing are in ../training/README.md. This directory includes probability records, not source photographs.

## Files

- fig0_system_architecture: current local three-node architecture.
- fig1_confusion_matrix: baseline and fine-tuned counts.
- fig2_precision_recall_f1: class metrics with test support.
- fig3_roc_pr_curves: empirical one-vs-rest curves.
- fig4_training_history: training loss and validation macro F1.
- fig5_dataset_composition: split and product counts.
- fig6_baseline_comparison: test accuracy and macro F1.
- fig7_recall_tradeoffs: per-class baseline/after-training recall.

Source metrics: ../training/results/2026-10-28/metrics.json.
