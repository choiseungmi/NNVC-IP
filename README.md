# NNVC-IP
경희대학교 컴퓨터공학과 캡스톤디자인2    
주제: Neural Network based Intra Prediction for post-VVC

Team
-------------
#### professor
김휘용

#### Student 
2018102242	최승미	<2018102242@khu.ac.kr>    

Our Test Sequence
-------------
#### The Common Test Conditions(CTC) of the test sequence we used are as follows.
- Chroma Format: YUV 4:2:0
- Input bit-depth: 8
- VVC QP: 22, 27, 32, 37

#### VVC test Sequence
Class C, D

#### CLIC 2020
Train, Test, Valid

Reference Software
-------------
VVC (VTM 9.0):
[Download VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tags)
[Download Documents](https://jvet.hhi.fraunhofer.de/)

Our Proposed Method
-------------
### Process

### Diagram

Performance Test
-------------

Code explanation
-------------
#### The Directory Structure
<span style="color:yellow">local directory settings</span>    
public directory settings   
```
.
├── README.md
├── .gitignore
├── VTM               // reference software VTM 9.0
|   ├── <span style="color:yellow">bin</span>                     // VTM 실행 폴더
|   |   ├── <span style="color:yellow">input</span>               // VTM input
|   |   ├── <span style="color:yellow">output</span>              // VTM output for traind dataset
|   |   ├── <span style="color:yellow">output_anchor</span>       // VTM output for inference dataset (predictor 신호 포함)
|   |   ├── <span style="color:yellow">EncoderApp.exe</span>      // VTM Encoder
|   |   ├── <span style="color:yellow">DecoderApp.exe</span>      // VTM Decoder
|   |   ├── <span style="color:yellow">encoder_intra_vtm(and train).cfg </span>     // cfg file
|   |   └── <span style="color:yellow">*.bat</span>               // VTM run bat files
|   └── ...
├── checkpoint        // models folder
├── <span style="color:yellow">train_data</span>        // 전처리를 마친 train data folder
├── <span style="color:yellow">valid_data</span>        // 전처리를 마친 validation data folder
├── <span style="color:yellow">anchor_data</span>       // 전처리를 마친 inference data folder
├── <span style="color:yellow">runs</span>              // train tensorboard log folder
├── <span style="color:yellow">log_csv</span>           // train csv log folder
├── dataset
|   ├── load_dataset.py   // Custom Dataset code
|   └── make_dataset.py   // Preprocessing and make dataset code
├── models
|   ├── __init__
|   ├── alexnet.py        // alexnet code
|   ├── tapnn.py          // proposed model code
|   └── vgg16.py          // vgg16 code
├── <span style="color:yellow">config.py</span>                 // total config file
├── inference.py              // inference and test code
├── make_CLIC_VTM_bat.py      // make vtm run bat file code for Image Dataset
├── make_CTC_VTM_bat.py       // make vtm run bat file code for Test sequence 
├── train.py              // proposed model train code
├── train_cluster.py      // cluster model train code
└── util.py               // utils code
```   

#### 명령어
```python
python train.py --epochs [epochs] -lr [learning rate] --batch-size [batch size] -hgt [block height] - wdt [block width] -q [quality] --clusterk [index of cluster] --cuda --save    
```
```python
python train_cluster.py --epochs [epochs] -lr [learning rate] --batch-size [batch size] -hgt [block height] - wdt [block width] -q [quality] --cuda --save     
```
