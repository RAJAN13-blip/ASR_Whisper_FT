from whisper import WhisperForConditionalGeneration, processor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from whisper import data_collator
from metrics import compute_metrics, metric
import hydra
from hydra.core.config_store import ConfigStore
from data import prepare_data
from functools import partial
import logging
from config import trainingArguments
import os
from prep import StreamToLogger
import sys

log = logging.getLogger(__name__)
cs = ConfigStore.instance()
cs.store(name = "object_composition_config", node = trainingArguments)
otrain, oval = prepare_data(test=False,
                            AUDIO_COLUMN_NAME = "file_name",
                            TEXT_COLUMN_NAME = "phrase",
                            INTENT_COLUMN_NAME = "prompt")

@hydra.main(config_path="config",config_name="trainconfig")
def main(cfg :trainingArguments):


    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
    store_path = cfg.store_path
    confg_name = cfg.configname

    ckpt_path = store_path +  confg_name
    try:
        os.mkdir(ckpt_path)
    except:
        print("config already exists")

    # Setting up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=ckpt_path,  # change to a repo name of your choice
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=float(cfg.learning_rate),
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=150,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        logging_dir=log.root.handlers[1].baseFilename.rsplit("/",1)[0]
    )

    # Set up trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=otrain,
        eval_dataset=oval,
        data_collator=data_collator,
        compute_metrics= partial(compute_metrics,processor=processor, metric=metric),
        tokenizer=processor.feature_extractor,
    )


    # Ignition !!
    log.info(trainer.state.log_history)
    trainer.train()

    # Save the model and processor 
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()