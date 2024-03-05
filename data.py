from prep import *
from whisper import processor

overview_path = "Medical Speech, Transcription, and Intent/overview-of-recordings.csv"
base_path = "Medical Speech, Transcription, and Intent/recordings/"

# Get train test and val files 
train_files = os.listdir(base_path + 'train')
val_files = os.listdir(base_path + 'validate')
test_files = os.listdir(base_path + 'test')

def prepare_data(overview_path=overview_path,test=False,**kwargs):

# Prepare seperate Dataframes for Audio2Text and Text2Text
    overview = pd.read_csv(overview_path)
    overview = overview[[kwargs["AUDIO_COLUMN_NAME"],kwargs["TEXT_COLUMN_NAME"],kwargs["INTENT_COLUMN_NAME"]]]

    overviewAudio = overview[[kwargs["AUDIO_COLUMN_NAME"],kwargs["TEXT_COLUMN_NAME"]]]
    overviewText = overview[[kwargs["TEXT_COLUMN_NAME"],kwargs["INTENT_COLUMN_NAME"]]]

    # Get OverviewAudio files 
    oatrain = overviewAudio[overview.file_name.isin(train_files)]
    oaval = overviewAudio[overview.file_name.isin(val_files)]
    

    def prepare_dataset(batch, split, processor=processor):
        # load and resample audio data frotestm 48 to 16kHz
        path = base_path + split + '/' +  batch["file_name"]
        audio,rate = librosa.load(path, sr=16000)

        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(audio, sampling_rate=rate).input_features[0]

        # encode target text to label ids
        batch["labels"] = processor.tokenizer(batch["phrase"]).input_ids
        return batch

    prep_train = lambda x : prepare_dataset(x,"train")
    oatrain = oatrain.apply(prep_train, axis=1)

    # Get val datasets
    prep_val = lambda x : prepare_dataset(x,"validate")
    oaval = oaval.apply(prep_val, axis=1)

    # Get test datasets 
    otest= None
    if test :
        oatest = overviewAudio[overview.file_name.isin(test_files)]
        prep_test = lambda x : prepare_dataset(x,"test")
        oatest = oatest.apply(prep_test, axis=1)
        otest = np.array(oatest[["input_features","labels"]])


    otrain = np.array(oatrain[["input_features","labels"]])
    oval = np.array(oaval[["input_features","labels"]])

    return otrain, oval

