# Architecture chapters

## MLP (`dl_a_mlp`)

Family: dense. The model returns one fraud logit and exposes auxiliary representations for XAI. Its model state and epoch telemetry are saved under evaluation2/ and checkpoints2/.

## TabNet-style attentive network (`dl_b_tabnet`)

Family: attention. The model returns one fraud logit and exposes auxiliary representations for XAI. Its model state and epoch telemetry are saved under evaluation2/ and checkpoints2/.

## 1D convolutional tabular network (`dl_c_cnn1d`)

Family: convolution. The model returns one fraud logit and exposes auxiliary representations for XAI. Its model state and epoch telemetry are saved under evaluation2/ and checkpoints2/.

## Autoencoder anomaly hybrid (`dl_d_autoencoder`)

Family: anomaly. The model returns one fraud logit and exposes auxiliary representations for XAI. Its model state and epoch telemetry are saved under evaluation2/ and checkpoints2/.

## Feature-token transformer (`dl_e_transformer`)

Family: transformer. The model returns one fraud logit and exposes auxiliary representations for XAI. Its model state and epoch telemetry are saved under evaluation2/ and checkpoints2/.
