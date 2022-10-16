# RNABERT
This repo contains the code for the paper "Informative RNA-base embedding for functional RNA clustering and structural alignment". You can contact the author at akiyama@dna.bio.keio.ac.jp for any question. Please cite this paper if you use the code or system output. 
Here it was strip of all the code that permit to train, align and cluster. It was also modified to be able to read the pre-train model on a cpu only.

## 1. Environment setup

The code is written with python Python 3.6.5. Our code requires PyTorch version >= 1.4.0, biopython version >=1.76, and C++17 compatible compiler. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.


#### 1.1 Install the package and other requirements

(Required)

```
python setup.py install
```

#### 2.1 Download pre-trained RNABERT

[RNABERT](https://drive.google.com/file/d/1sT6jlv9vrpX0npKmnbFeOqZ1JZDrZTQ2/view?usp=sharing)

Download the pre-trained model in to a directory. 
This model has been created using a full Rfam 14.3 dataset (~400nt). 

## 3. Earn embeddings

To obtain the embedding vector for the RNA sequence, run 

```
python MLM_SFP.py 
    --pretraining ${PRE_WEIGHT} \
    --data_embedding ${PRED_FILE} \
    --embedding_output ${OUTPUT_FILE} \
    --batch 40 \
```