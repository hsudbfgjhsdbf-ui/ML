# Approach 2 overview

Approach 2 extends the traditional baseline with five deep tabular architectures
and first-class XAI. It reuses the supplied workbook and split fingerprint
`3bc2230ff033d29d4eb664ca8b78f2c902416a154bcc0cee88d0863881d59c9f`. The validation-selected deep model is **Feature-token transformer** with mean validation PR-AUC **0.9800** and mean F2 **0.9709**.

The run is laptop-first and deliberately does not claim a 50-trial Bayesian
search. Three seeds, checkpointed epochs, common metrics, occlusion, deletion
faithfulness, stability, calibration, and fairness provide an honest reference.
