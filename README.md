
# wmt24-lowres-spain
We participated in the *WMT24 shared task for Low-Resource Languages of Spain* under the **Constrained Submission** task
Team Name : Mora-translate
Primary submission id: 547

This code repo contains the source code and data used in the paper **"Back to the Stats: Rescuing Low Resource Neural Machine Translation with Statistical Methods"**

## Data
Training is conducted in two steps: first, training the entire model using the filtered Spanish-Asturian CCMatrix dataset; then, fine-tuning the best model from the training phase by freezing the encoder layers.
Data for training step 1 and 2 are available in their respective folder in [here](https://github.com/vmenan/wmt24-lowres-spain/tree/main/data).

## Data Filtration code
Please visit the folder [`data_filtration_code`](https://github.com/vmenan/wmt24-lowres-spain/tree/main/data_filtration_code) for the codes utilized in this pipeline.

## Training
Training is conducted in two steps: first, training the entire model using the filtered Spanish-Asturian CCMatrix dataset; then, fine-tuning the best model from the training phase by freezing the encoder layers. Code for both steps are provided as individual python scripts in the `training_code` folder.
