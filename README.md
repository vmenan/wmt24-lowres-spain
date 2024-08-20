
# wmt24-lowres-spain
We participated in the *WMT24 shared task for Low-Resource Languages of Spain* under the **Constrained Submission** task
Team Name : Mora-translate
Primary submission id: 547

This code repo contains the source code and data used in the paper **"Back to the Stats: Rescuing Low Resource Neural Machine Translation with Statistical Methods"**

**All code is written in Python programming language**

## Data
Training is conducted in two steps: first, training the entire model using the filtered Spanish-Asturian CCMatrix dataset; then, fine-tuning the best model from the training phase by freezing the encoder layers.
Data for training step 1 and 2 are available in their respective folder in [here](https://github.com/vmenan/wmt24-lowres-spain/tree/main/data).

## Data Filtration code
Please visit the folder [`data_filtration_code`](https://github.com/vmenan/wmt24-lowres-spain/tree/main/data_filtration_code) for the codes utilized in this pipeline.

## Training
Training is conducted in two steps: first, training the entire model using the filtered Spanish-Asturian CCMatrix dataset; then, fine-tuning the best model from the training phase by freezing the encoder layers. Code for both steps are provided as individual python scripts in the [`training_code`](https://github.com/vmenan/wmt24-lowres-spain/tree/main/training_code) folder.

We use the `meta_data.json` file to provide the all the hyperparameters for the training. Both codes used for step 1 (`small100_training.py`) and step 2 (`small100_finetuning.py`) use the same `meta_data.json` file. 

**IMPORTANT!**: When using the `meta_data.json` file for step 1, in the arguments `model_checkpoint` you must provide the original pretrained checkpoint available in HuggingFace found here. And when you are using the step 2 code for finetuning the model from step 1, provide the path to model saved from step 1 to the `model_checkpoint` argument in the `meta_data.json` file.

The best performing model from our experiments can be downloaded from [here](https://drive.google.com/file/d/1YKKe6CI8tJAUAMWfuiLbDEloyPKsfg3w/view?usp=sharing).
