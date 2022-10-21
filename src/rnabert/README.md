# <center> 🦙RNABERT🦙</center>

> This repo contains the code for the paper **"Informative RNA-base embedding for functional RNA clustering and structural alignment"**. You can contact the author at akiyama@dna.bio.keio.ac.jp for any question. Please cite this paper if you use the code or system output. 

Here it was strip of all the code that permit to train, align and cluster. It was also modified to be able to read the pre-train model on a cpu only.

## 1. Environment setup 🔧

The code is written with python Python 3.6.5. Our code requires `PyTorch >= 1.4.0` and `biopython version >= 1.76`. It will onlye read the pretrain model given here.


**⚠️ Require:** Use the environment in (from the directory `rnabert`) `../../env` and type those commands:

```
mamba env create --file env_RNABERT_emb.yaml
conda env create --file env_RNABERT_emb.yaml
conda activate env_RNABERT_emb.yaml
```

## 2 Download pre-trained RNABERT 📶

Download the pre-trained model in to a directory here: [RNABERT](https://drive.google.com/file/d/1sT6jlv9vrpX0npKmnbFeOqZ1JZDrZTQ2/view?usp=sharing). 
This model has been created using a full `Rfam 14.3` dataset (~400nt). 

## 3. Embeddings ⚒️

To obtain the embedding vector for the RNA sequence, give a RNA in a fasta and run:

```
python MLM_SFP.py 
    --pretraining ${PRE_WEIGHT} \
    --data_embedding ${PRED_FILE} \
    --embedding_output ${OUTPUT_FILE} \
    --batch 40 \
```
## 4. Download the embeddings 📶

Here is the link to download the embedding that we obtain: https://drive.google.com/drive/folders/1wAsXxWd5TrJs1cu3K-GJJgGXHrTv_EL3?usp=sharing
