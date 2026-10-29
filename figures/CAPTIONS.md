# Journal figure captions

**fig0_system_architecture** — Implemented local workflow. Gateway applies deterministic rules; the classifier provides product-type advice only. T4 values above USD 2,500 require a human decision. Arrows are architecture, not measured telemetry.

**fig1_confusion_matrix** — Counts from the retained 145-image test split; rows are source labels and columns predicted hardware types. Neither matrix evaluates damage or return decisions.

**fig2_precision_recall_f1** — Per-class precision, recall and F1 on 145 images. GPU recall is 52.17%; SSD 19/19 is a small, source-limited sample and does not establish general perfect performance.

**fig3_roc_pr_curves** — Empirical one-vs-rest hardware-type ROC and precision-recall curves computed from stored probabilities for the same 145 test images. These are not damage-detection curves. Average precision uses recall-increment weighting; no confidence intervals or independent replication are claimed.

**fig4_training_history** — Twelve recorded training epochs; checkpoint selected by validation macro F1 at epoch 9. Curves are training loss and validation F1, not training/test accuracy.

**fig5_dataset_composition** — 971 original retained images after duplicate auditing: 679 train, 147 validation and 145 test. Image groups use exact-pixel deduplication and pHash <= 6; physical-item independence is not established.

**fig6_baseline_comparison** — Fine-tuned recognition accuracy 122/145 (84.14%) versus frozen-feature centroid baseline 109/145 (75.17%); macro F1 0.8301 versus 0.7339. This baseline is not Qwen3-VL or the old RMA system; GPU recall declined after training.

**fig7_recall_tradeoffs** — Recall changes on identical test images. GPU recall drops from 82.61% to 52.17%, while RAM recall rises from 51.06% to 85.11%. Report the tradeoff alongside aggregate accuracy.
