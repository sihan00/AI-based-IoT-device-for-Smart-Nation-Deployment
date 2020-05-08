#!/usr/bin/env python
# coding: utf-8

# # with adaptation, 
# https://www.swharden.com/wp/2016-07-19-realtime-audio-visualization-in-python/

import pyaudio
import time
# import pylab
import numpy as np
import mel_features
# from librosa.feature import librosa_lite

import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='tflite_mobilenet_2_edgetpu.tflite', experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """

    def __init__(self,device=None,startStreaming=True):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 32768 # 4096 # number of data points to read at a time
        self.rate = 44100 # time resolution of the recording device (Hz)

        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength=2 #seconds
        self.tape=np.zeros(self.rate*self.tapeLength) 
        # (np.empty(44100*2)*np.nan).shape
        # print(self.tape.shape) # (88200,)


        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()

    ### LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def stream_read(self):
        """return values for a single chunk"""
        data = np.frombuffer(self.stream.read(self.chunk),dtype=np.int16)
        #print(data)
        return data

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,
                                rate=self.rate,input=True,
                                frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    ### TAPE METHODS
    # tape is like a circular magnetic ribbon of tape that's continously
    # recorded and recorded over in a loop. self.tape contains this data.
    # the newest data is always at the end. Don't modify data on the type,
    # but rather do math on it (like FFT) as you read from it.

    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk]=self.tape[self.chunk:]
        self.tape[-self.chunk:]=self.stream_read()

    def tape_flush(self):
        """completely fill tape with new data."""
        readsInTape=int(self.rate*self.tapeLength/self.chunk) 
        print(" -- flushing %d s tape with %dx%.2f ms reads"%                  (self.tapeLength,readsInTape,self.chunk/self.rate))
        for i in range(readsInTape):
            self.tape_add()
            
            
    def tape_forever(self,plotSec=.25):
        t1=0
        try:
            while True:
                self.tape_add()
                
                # add stuff here
                # self.tape_logspec() 
                # add stuff here
                # self.tape_spec() = librosa.feature.melspectrogram(self.tape_add(), sr = self.rate, n_mels=128)
                # self.tape_logspec() = librosa.amplitude_to_db(self.tape_spec())
                # melspec = librosa.feature.melspectrogram(data.astype(float), sr=RATE, n_mels=128)
                # log_S = librosa.amplitude_to_db(melspec)
                
                if (time.time()-t1)>plotSec:
                    t1=time.time()
                    print()
                    # self.tape_plot()
                    # self.tape_processing()
                    self.tape_inferencing()
    
        except:
            print(" ~~ exception (keyboard?)")
            raise
            return

        
    def tape_inferencing(self):
        """inferencing content in the tape"""
        # print("running tape inferencing")
        
        self.log_S = mel_features.log_mel_spectrogram(self.tape/65536)
        #self.melspec = librosa_lite.melspectrogram(self.tape/65536, sr= self.rate, n_mels=128)
        #self.log_S = librosa_lite.amplitude_to_db(self.melspec)

        start_index_x = (224-128)//2
        # start_index_x = 48
        end_index_x = (224-128)//2 + 128
        # end_index_x = 176
        start_index_y = (224-169)//2
        # start_index_y = 25
        end_index_y = (224-169)//2 + 169
        # end_index_y = 198
        # print(start_index_x, end_index_x, start_index_y, end_index_y)

        # self.test_recorded_data = np.empty(shape=(1,224,224,3))
        # for i in range(number):
        # mean = np.mean(splits_pad[i][self.log_S]) # shape is alr (128,173)
        # data_squeeze = np.squeeze(train_all[i])# (128,173,1) to (128,173) 
        
        self.mean = np.mean(self.log_S) # shape is alr (128,173)
        self.spectrogram = self.mean*np.ones(shape = (224,224,3))
        for j in range(3):
            self.spectrogram[start_index_x:end_index_x, start_index_y:end_index_y,j] = self.log_S

        #self.test_recorded_data = self.spectrogram
        
        t3 = time.time()
        
        # x = np.expand_dims(test_converted[100], axis=0)
        # x = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        #input_data = x # shape (1,224,224,3)
        
        self.input_data = np.expand_dims(self.spectrogram, axis=0)
        self.input_data = np.array(self.input_data, dtype=np.uint8)
        self.input_tensor = interpreter.set_tensor(input_details[0]['index'], self.input_data)
        self.prediction = interpreter.invoke()

        def classify(i):
                    switcher = {
                        0: "fall",
                        1: "cough",
                        2: "shout",
                        3: "speech"
                    }
                    return switcher.get(i, "Invalid class")

        self.output_tensor = interpreter.get_tensor(output_details[0]['index'])
        print("inferencing took:                         %.02f ms"%((time.time()-t3)*1000))
        
        # print('sound class:', classify(np.argmax(self.output_tensor)))
        # print(self.output_tensor) # [[8 77 2 169]]
        classlab = self.output_tensor[0]
        classidx = np.argmax(classlab) 
    
        print("sound class:                             ", classify(classidx), classlab[classidx])
        print("all prediction outputs:                  ", classlab)
        
        
#         self.predictions = loaded_model.predict(np.expand_dims(self.spectrogram, axis=0))
#         print("inferencing took:                         %.02f ms"%((time.time()-t3)*1000))
    
#         # self.predictions = loaded_model.predict(self.spectrogram)
        
#         # print(np.argmax(self.predictions), self.predictions)
        
        
#         classidx = np.argmax(self.predictions[0]) 
        
#         def classify(i):
#             switcher = {
#                 0: "fall",
#                 1: "cough",
#                 2: "shout",
#                 3: "speech"
#             }
#             return switcher.get(i, "Invalid class")
        
#         classlab = self.predictions[0]
        
#         thresh = 0.60
#         if classlab[0] < thresh and classlab[1] < thresh and classlab[2] < thresh and classlab[3] < thresh:
#             print("sound class:                              ambience sound")
#         else:
#             print("sound class:                             ", classify(classidx), classlab[classidx])
    
        
#         # print("print all for checking:                  ", classidx, classlab)
        
#         print(np.argmax(self.predictions), self.predictions)
        


if __name__=="__main__":
    ear=SWHear()
    ear.tape_forever()
    ear.close()
    print("DONE")


