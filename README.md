# AI-based-IoT-device-for-Smart-Nation-Deployment
Si Han's Final Year Project, NTU EEE, AY2019/20 (A3089-191)

A project to implement a real-time audio classification system on the new platform of Google Coral.
Classify 4 classes of audio: fall, cough, shout, speech, (ambience). 
Released in March 2019, the Dev Board has an embedded Edge tensor processing unit (TPU) machine learning chip.

## Procedures:

1. Data pre-processing
2. Model training, conversion and compiling
3. Setting up the Coral Dev Board
4. Live audio acquisition and inferencing system on PC and Coral Dev Board

## Oweing to librosa library installation issues on Coral Dev Board:

a. coded own mel_features.py algorithm for audio features extraction, independent of librosa (not accurate)
b. prepared own librosa_lite.py and library which extracts only the files of librosa functions I need (accurate)

## Conclusion

Implementation was feasible and successful, though met with many compatibility errors with hardware. 
Most available documentation are image-based projects, my project expands existing applications. 
