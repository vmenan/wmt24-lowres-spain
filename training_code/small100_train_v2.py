import torch
import time
from datasets import  load_metric,Dataset
from transformers import M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tokenization_small100 import SMALL100Tokenizer
from transformers import DataCollatorForSeq2Seq
import json





def calculate_elapsed_time(start_time):
    # Get the current time
    end_time = time.time()

    # Calculate the time difference
    time_difference = end_time - start_time

    # Convert seconds to days, hours, minutes, and seconds
    days, remainder = divmod(time_difference, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return int(days), int(hours), int(minutes), int(seconds)





def train_and_save (meta_dict,
                    trainer):
    startTime=time.time()



    tokenizer.save_pretrained(meta_dict["tokenizer_ouput_folder"])
    trainer.train()
    trainer.save_model(meta_dict["model_output_folder"]+ "_final_model")

    days, hours, minutes, _ = calculate_elapsed_time(startTime)
    print('Time taken for training : {} days {} hrs {} mints'.format(days, hours, minutes))
    print('training finished')


def intialize_trainer(meta_dict,
                      tokenized_datasets_train,
                      tokenized_datasets_test,
                      data_collator):
    #initialized model parameters
    args = Seq2SeqTrainingArguments(output_dir=meta_dict["model_output_folder"],
                            do_train=True,
                            do_eval=True,
                            per_device_train_batch_size=int(meta_dict["per_device_train_batch_size"]),
                            per_device_eval_batch_size=int(meta_dict["per_device_eval_batch_size"]),
                            learning_rate=float(meta_dict["learning_rate"]),
                            weight_decay=float(meta_dict["weight_decay"]),
                            save_total_limit=1,
                            num_train_epochs=int(meta_dict["num_train_epochs"]),
                            gradient_accumulation_steps = int(meta_dict["gradient_accumulation_steps"]),
                            predict_with_generate=True,
                            fp16=True,
                            eval_accumulation_steps=1,
                            save_strategy = "epoch",
                            evaluation_strategy="epoch",
                            load_best_model_at_end = True,
                            push_to_hub=False
                            )


    #initlizing trainer
    trainer = Seq2SeqTrainer(model=model,
                    args=args,
                    data_collator=data_collator,
                    train_dataset=tokenized_datasets_train,
                    eval_dataset=tokenized_datasets_test)
    return trainer

def preprocess_function(examples):
    source = meta_dict["src_lang"]
    target = meta_dict["tgt_lang"]
    inputs = [ex[source] for ex in examples["translation"]]
    targets = [ex[target] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=int(meta_dict["max_length"]), padding=True, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=int(meta_dict["max_length"]) , padding=True, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




def prepare_data(meta_dict):
    #training data
    train_data=[]
    with open(meta_dict["train_source_path"]) as f1, open(meta_dict["train_target_path"]) as f2:
        for src, tgt in zip(f1, f2):
            train_data.append(
                {
                    "translation": {
                        meta_dict["src_lang"]: src.strip(),
                        meta_dict["tgt_lang"]: tgt.strip()
                    }
                }
            )
    print(f'total size of train data is {len(train_data)}')

    #Validationi data
    valid_data=[]
    with open(meta_dict["valid_source_path"]) as f1, open(meta_dict["valid_target_path"]) as f2:
        for src, tgt in zip(f1, f2):
            valid_data.append(
                {
                    "translation": {
                        meta_dict["src_lang"]: src.strip(),
                        meta_dict["tgt_lang"]: tgt.strip()
                    }
                }
            )
    print(f'total size of train data is {len(valid_data)}')
    training_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(valid_data)
    tokenized_datasets_train = training_dataset.map(preprocess_function, batched=True)
    tokenized_datasets_test = test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return tokenized_datasets_train,tokenized_datasets_test,data_collator





def intialize_model(meta_dict):
    torch.cuda.empty_cache()
    #model checkpoint
    model_checkpoint = meta_dict["model_checkpoint"]
    tokenizer_checkpoint = meta_dict["tokenizer_checkpoint"]

    #initialize metric
    metric = load_metric("sacrebleu")

    #initializing Tokenizer & Model
    model = M2M100ForConditionalGeneration.from_pretrained(model_checkpoint).to("cuda")
    tokenizer = SMALL100Tokenizer.from_pretrained(tokenizer_checkpoint,
                                                tgt_lang=meta_dict["tgt_lang"])
    return metric,model, tokenizer

def read_meta(json_file_path):
    with open(json_file_path, 'r') as file:
        model_metadata = json.load(file)
    return model_metadata

if __name__ == "__main__":
    global meta_dict 
    meta_dict = read_meta("meta_data.json")
    # print(meta_dict)
    metric,model, tokenizer = intialize_model(meta_dict)
    tokenized_datasets_train,tokenized_datasets_test,data_collator = prepare_data(meta_dict)
    trainer = intialize_trainer(meta_dict,
                      tokenized_datasets_train,
                      tokenized_datasets_test,
                      data_collator)
    train_and_save (meta_dict,
                    trainer)