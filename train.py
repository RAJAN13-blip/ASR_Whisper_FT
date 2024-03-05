from whisper import WhisperForConditionalGeneration, processor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from whisper import data_collator
from metrics import compute_metrics, metric
from data import prepare_data
from functools import partial


otrain, oval = prepare_data(test=False,
                            AUDIO_COLUMN_NAME = "file_name",
                            TEXT_COLUMN_NAME = "phrase",
                            INTENT_COLUMN_NAME = "prompt")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

# Setting up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medical-data",  # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=500,
    gradient_checkpointing=True,
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

model.save_pretrained(training_args.output_dir)
if __name__ == "__main__":
    # Ignition !!
    trainer.train()

    # Save the model and processor 
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)