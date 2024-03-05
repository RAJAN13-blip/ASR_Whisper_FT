from data import prepare_data
from whisper import data_collator
from whisper import WhisperForConditionalGeneration, processor
import os 

model  = WhisperForConditionalGeneration.from_pretrained("whisper-medical-data")

otrain, oval, otest = prepare_data(test=True,
                                   )
op = data_collator(oval)
# load dummy dataset and read audio files
data_val_samples = data_collator(oval[:20])
input_features = data_val_samples.input_features.cuda()

# generate token ids
label_ids = processor.batch_decode(oval[:20,1],skip_special_tokens=True)
predicted_ids = model.generate(input_features)

# decode token ids to text
transcription_t = processor.batch_decode(predicted_ids, skip_special_tokens=True)
