# Approach 2 speaker notes

## Slide 1

Introduce Approach 2 as a controlled extension of the traditional baseline. The same rows, split, target semantics, and metric canon are reused; only the representation and optimization family changes.

## Slide 2

Position the deep approach as a scientific comparison, not a claim that neural networks are automatically better on tabular data.

## Slide 3

Explain that the hypotheses can be confirmed, refuted, or inconclusive. The point of the run is evidence, not a predetermined win.

## Slide 4

This is the key methodological slide. It prevents a deep model from benefiting from a different split or a more favorable target definition.

## Slide 5

The architectures intentionally cover diverse inductive biases. All emit one fraud logit so the shared evaluator can compare probabilities.

## Slide 6

Walk the tensor contract and mention that categorical embeddings are a future extension because the comparison uses the frozen Approach 1 matrix.

## Slide 7

Masks are useful for interpretation but are not automatically explanations; faithfulness is checked with the same occlusion framework.

## Slide 8

This is a good examiner question. Tabular columns do not have a natural spatial order; the architecture is included as a falsifiable probe, not as an unquestioned assumption.

## Slide 9

The autoencoder is the only hybrid architecture. Its reconstruction loss is auxiliary and does not change the fraud-positive target semantics.

## Slide 10

Mention the distinction between attention visualization and explanation faithfulness. The scored explanation is model-agnostic occlusion.

## Slide 11

The telemetry is evidence about training dynamics. It helps diagnose overfitting and instability rather than just reporting the final score.

## Slide 12

Read the leaderboard as mean validation evidence. The standard deviation is shown so a small apparent gain with unstable seeds is not over-celebrated.

## Slide 13

Use the learning curve to discuss early stopping and train-validation gaps. The best epoch is restored, not the final epoch.

## Slide 14

PR curves are more diagnostic at six-percent prevalence. ROC curves remain useful for broad ranking but should not be the only result.

## Slide 15

Calibration helps interpret a probability, but does not remove distribution shift. The threshold is still a workflow policy that needs business and regulatory review.

## Slide 16

The XAI design prevents architecture marketing from dominating the score. All five models receive the same primary explanation contract.

## Slide 17

Explain the meaning of a positive occlusion change: removing the feature lowered the model probability on the sampled rows. It is not proof that the underlying claim detail caused fraud.

## Slide 18

Deep models can reproduce or amplify shortcut signals. Fairness is checked at the decision layer and should be repeated on a larger validated dataset.

## Slide 19

The reference run keeps augmentation off for attribution clarity. Future variants should be labelled non-reference and evaluated with the same protocol.

## Slide 20

The deep approach is valuable even if it does not beat the classical winner. It answers when learned representations justify their additional cost.

## Slide 21

Be direct about deviations from the full prompt. The project avoids fabricated deep metrics and labels the current run as a laptop-first reference benchmark.

## Slide 22

The agentic approach should not erase the tabular evidence; it should add documents, policy reasoning, and structured collaboration.

## Slide 23

Close with the evidence culture: deep learning adds capacity, but trustworthy fraud screening still requires validated data, calibrated probabilities, explanations, and human decisions.
