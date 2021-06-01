# Lexical Stress Detection for isolated English words

Author: Vera Bernhard

Date: 01.06.21

## Requirements

- Praat == 6.1.39

- Prosogram == v300 (July 15, 2020)

- Python >= 3.8

- Python specific requirements can be found in:
  `requirements.txt`

## 1. Some Functionalities of `StressDetector`

- 3 `.wav` files are provided in `/wav_tg_all`. To test on the entire dataset, copy all .wav files from the Switch Drive into the folder `/wav_tg_all`.

- The following commands need to be run within directory `/stress-detector`.

- Adapt the Prosogram path in `scripts/praat/run_prosogram.praat` matching your environment.

### Load pretrained model and classify

```python
from stress_detector import StressDetector, FEATURES
wav_path = './test'
sd = StressDetector(wav_path, FEATURES)
sd.load_classifier('models/classifier_vot_210601.pkl', 'models/scaler_vot_210601.pkl')
print(sd.classify('test/bamboo1.wav', 'bamboo'))
# >>> ([0, 1], [0, 1]) # True: [0, 1], Pred: [0, 1]
print(sd.classify('test/bamboo2.wav', 'bamboo'))
# >>> ([0, 1], [1, 0]) # True: [0, 1], Pred: [1, 0]
sd.classify('test/bamboo2.wav', 'bamboo', feedback=True)
# >>>   stressed incorrectly:
#       stress in first syllable "b { m"
#       instead of second syllable "b u:"
```

### Preprocess all data

- Creates syllable alignment by using WEBMAUS, runs PROSOGRAM on all data

- `wav_path`: directory containing all .wav files, named in form S1_LOC_2_1_alarm_1.wav

- `par_path`: directory with all phonetic transcription of training words in .par format

- Files created by WEBMAUS or PROSOGRAM are all saved within `wav_path`

#### Variant 1: Webmaus and Prosogram needs to be run, only .wav given

```python
from stress_detector import StressDetector, FEATURES

wav_path = './wav_tg_all'
par_path = './photrans'
sd = StressDetector(wav_path, FEATURES)
sd.preprocess(par_path)

```

#### Variant 2: Only Prosogram needs to be run , .wav & .TextGrid given

```python
from stress_detector import StressDetector, FEATURES

wav_path = './wav_tg_all'
sd = StressDetector(wav_path, FEATURES)
sd.preprocess()

```

### Calculate features of preprocessed data

Features have not been calculated yet but .TextGrid and Prosogram output are given

```python
from stress_detector import StressDetector, FEATURES

wav_path = './wav_tg_all'
sd = StressDetector(wav_path, FEATURES)
sd.read_in()
sd.get_features().to_csv('./data/complete_features_subset.tsv', sep='\t')
```

### Check in how many files Prosogram has found all syllables

```python
from stress_detector import StressDetector, FEATURES

wav_path = './wav_tg_all'
sd = StressDetector(wav_path, FEATURES)
sd.read_in()
sd.get_features('./data/complete_features_subset.tsv')
print(sd.agree_syllable_nr())
```

### Evaluate a classifier on all data

```python
from stress_detector import StressDetector, FEATURES
from sklearn.tree import DecisionTreeClassifier
import numpy as np

wav_path = './wav_tg_all'
sd = StressDetector(wav_path, FEATURES)
sd.get_features('./data/complete_features.tsv')
clf = DecisionTreeClassifier()
evaluation = sd.train(clf)
print('F1 Score: {}'.format(np.mean(evaluation['f1'])))
```

## 2. Finding the best classifier

The process of finding the best classifier is documented in `find_model.py`

The final Voting Classifier trained in this thesis and the corresponding scaler are saved in `models\`:
* `classifier_vot_210601.pkl`
* `scaler_vot_210601.pkl`
