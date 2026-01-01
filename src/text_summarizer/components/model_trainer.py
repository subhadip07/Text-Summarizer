from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.text_summarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        ## Loading the Data
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Create TrainingArguments with compatibility for different transformers versions
        import inspect

        common_args = dict(
            output_dir='pegasus-samsum',
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            eval_steps=500,
            save_steps=int(1e6),
            gradient_accumulation_steps=16,
        )

        sig_params = inspect.signature(TrainingArguments).parameters
        if 'evaluation_strategy' in sig_params:
            common_args['evaluation_strategy'] = 'steps'
        else:
            # Older transformers versions don't support `evaluation_strategy`; ensure evaluation runs
            common_args['do_eval'] = True

        trainer_args = TrainingArguments(**common_args)

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt['train'],
                  eval_dataset=dataset_samsum_pt['validation'])
        
        trainer.train()

        ## Save Model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))

        ## Save Tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
