원본 : https://github.com/Aubrey-ao/HumanBehaviorAnimation/tree/main/HumanBehaviorAnimation/RhythmicGesticulator/Simplified_Version



## Datset
[Trinity Speech-Gesture Dataset (GENEA Challenge 2020)](https://trinityspeechgesture.scss.tcd.ie/).


## Install

Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html.

``` shell
cd HumanBehaviorAnimation/RhythmicGesticulator/Simplified_Version
conda env create -f environment.yaml
conda activate rhyGes_simple
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

## Dataset

Download [Trinity Speech-Gesture Dataset (GENEA Challenge 2020)](https://trinityspeechgesture.scss.tcd.ie/), and put the dataset into folder {Data} like:

```
- Data
  - Trinity
    - Source
      - Training_Data
        - Audio
          - Recording_001.wav
            ...
        - Motion
          - Recording_001.bvh
            ...
      - Test_Data
        - Audio
          - TestSeq001.wav
            ...
        - Motion
          - TestSeq001.bvh
            ...
```

1. Data Preprocessing
  - run preprocess.py
    - ./Data_Preprocessing/Config/Trinity/config.json5 에서 설정값 변경 가능

2. Gesture Lexicon
  - run train.py
    - main에서 원하는 네트워크 부분 주석 해제해서 사용
    - checkpoint 이어서 학습하고싶을 경우 self.use_checkpoint = True 주석해제
    - loaded_log_dir에 원하는 checkpoint 폴더명 입력
      - checkpoint는 ./Gesture_Lexicon/Training/Trinity/에 생성
  - lexeme의 분포를 시각화하고 싶을 경우 run lexicon_VQVAE.py
    - checkpoint_path와 checkpoint_config에 원하는 checkopint 폴더명 입력

3. Gesture Generator
  - run train.py
    - ./Gesture_Generator/Config/Trinity/config.json5 에서 RNN hparams 사전 설정 필요
  - lexeme만을 이용한 inference를 원할 경우 run inference.py
  - 모든 lexeme별 inference를 원할 경우 run inference_forlexeme.py
  - motion, audio, lexeme을 이용한 inference를 원할 경우 inference.py에서 다음 주석 해제 필요
```
infer_res = infer_train(batch, self.device, self.net, uniform_len, num_blocks, self.config["network"]["name"])
```
