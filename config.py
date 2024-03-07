from dataclasses import dataclass 
from typing import Optional 


@dataclass
class trainingArguments:
    per_device_train_batch_size : int
    gradient_accumulation_steps : int
    learning_rate : float
    warmup_steps : int
    max_steps : int
    store_path : str
    configname : str
    gradient_checkpointing : Optional[bool] = True

