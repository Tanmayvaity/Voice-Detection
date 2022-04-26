import os.path
import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 16000
dataset_path = "dataset/speeches"
test_path = "test/BENJAMIN.wav"

def create_mapings(data_path):
    mapping = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):

        # ensure we're at sub-folder level
        if dirpath is not data_path:
            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1][9:]
            mapping.append(label)
            # print("\ndirectory: '{}'".format(label))

    return mapping


class Voice_Detection():


    model = None
    # _mapping = create_mapings(dataset_path)
    _mapping = [
        "Benjamin Netanyau",
        "Jens Stoltenberg",
        "Julia Gillard",
        "Magaret Tarcher",
        "Nelson Mandela"
    ]


    _instance = None





    def predict(self, file_path):


        # extract MFCC
        MFCCs = self.preprocess(file_path)
        # if MFCCs == -1 :
        #     return -1,-1,-1

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword,predictions,predicted_index


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):


        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        # else:
        #     return -1

        return MFCCs.T


def voice_detection_system():


    # ensure an instance is created only the first time the factory function is called
    if Voice_Detection._instance is None:
        Voice_Detection._instance = Voice_Detection()
        Voice_Detection.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return Voice_Detection._instance




if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = voice_detection_system()
    kss1 = voice_detection_system()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    speaker = kss.predict(test_path)
    print(speaker)