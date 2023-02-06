# MutFormer
MutFormer is an application of the BERT (Bidirectional Encoder Representations from Transformers) NLP (Natural Language Processing) model with an added adaptive vocabulary to protein context, for the purpose of predicting the effect of missense mutations on protein function.

For this project, a total of 6 models were trained:

Model Name | Hidden Layers | Hidden Size (and size of convolution filters) | Intermediate Size | Input length | # of parameters | Download link (right click mouse "save link as...")
-----------|---------------|-------------|-------------------|--------------|-----------------|--------------
MutBERT8L | 8 | 768 | 3072 | 1024 | ~58M | http://www.openbioinformatics.org/mutformer/MutBERT_8L.zip
MutBERT10L | 10 | 770 | 3072 | 1024 | ~72M | http://www.openbioinformatics.org/mutformer/MutBERT_10L.zip
MutFormer8L | 8 | 768 | 3072 | 1024 | ~62M | http://www.openbioinformatics.org/mutformer/MutFormer_8L.zip
MutFormer10L | 10 | 770 | 3072 | 1024 | ~76M | http://www.openbioinformatics.org/mutformer/MutFormer_10L.zip
MutFormer12L (Same size transformer as BERT-base) | 12 | 768 | 3072 | 1024 | ~86M | http://www.openbioinformatics.org/mutformer/MutFormer_12L.zip
MutFormer8L (integrated adap vocab) | 8 | 768 | 3072 | 1024 | ~64M | http://www.openbioinformatics.org/mutformer/MutFormer_emb_adap_8L.zip

MutBERT8L and MutBERT10L use the original BERT model for comparison purposes, the MutFormer models are the official models.

## To use MutFormer:

### Precomputed Scores

We have included precomputed scores for all known missense protein-altering mutations in the human proteome in the DBNSFP42 database (hg19 build). This has been included in both hg19 and hg39 coordinates as an asset file titled "hg19_mutformer.zip" and "hg38_mutformer.zip."

#### Alternatively, a direct link: http://www.openbioinformatics.org/mutformer/hg19_MutFormer.zip

One way to use these scores is through Annovar. Given a .avinput file with tab delimited:

1. chromosome number
2. mutation start position
3. mutation end position (will be the same as start position for SNPs)
4. reference sequence (nucleotide segment prior to mutation)
5. mutated segment (nucleotide segment after mutation)

Annovar's table_annovar.pl tool (https://annovar.openbioinformatics.org/en/latest/) can be used to analyze the mutations present in avinput file based on MutFormer scores. To use annovar on an avinput formatted file:

1. Fill out a registration form and download Annovar from the Annovar website: https://www.openbioinformatics.org/annovar/annovar_download_form.php

2. There will most likely be a “/humandb” in the resulting downloaded software folder. Download, unzip, and move  "hg19_mutformer.zip" into this humandb folder, and within the software folder, navigate to “annotate_variation.pl.”

3. Create a new conda environment, install perl, and run annotate_variation.pl to analyze an avinput file:

```
conda env create <desired env name>
conda activate <desired env name>
conda install perl
perl table_annovar.pl "<PATH_TO_INPUT_FILE>" humandb/ -protocol MutFormer -operation f -build hg19 -nastring . -out "<DESIRED_OUTPUT_PATH>" -polish
```

We have also included an example setup as a zip file: "annovar_mutformer_example.zip" with the release. To run this example, download and unzip the "hg19_mutformer.zip" scores file, and follow the instructions in the README.txt within the folder.

### Inference

To run the trained MutFormer model, one can choose to run MutFormer on a cloud TPU via Colab (using mutformer_finetuning/mutformer_finetuning_eval_predict.inpyb, specifying 'predict' mode in the Eval/prediction loops section), run inference locally, for which a python script is provided, or use a mutation annotation software such as annovar.

To run the trained MutFormer model, one can run inference locally using a python script provided:

#### Local Run

To run MutFormer for inference locally, download/clone the MutFormer repo and navigate via Terminal into the folder:  "mutformer/mutformer_inference/local_run." Create a conda environment using the yml file "mutformer_inference.yml" provided:

```
conda env create -f mutformer_inference.yml -n <desired env name>
```

Then, run the script "run_inference.py," specifying the parameters.

Parameters for run_inference.py include:

--input_file: Input raw text file for running inference (or comma-separated list of files) (should be formatted corresponding to the test mode chosen from the Finetuning section in the README).

--output_file: File to output inference results to (or comma-separated list of files).

--model_folder: Folder where the model checkpoint and config file is stored.

--model_architecture: Model architecture of the model checkpoint specified (BertModel indicates the original BERT, BertModelModified indicates MutFormer's architecture without integrated convs, MutFormer_embedded_convs indicates MutFormer with integrated convolutions).

--vocab_file: The vocabulary file that the BERT model was trained on.

--batch_size: Number of lines to batch together at a time when performing inference.

--max_seq_length: Maximum sequence length to pad to; should be greater than or equal to three more than twice the length of the longest input amino acid sequence.

--using_ex_data: set this to True if external data is being included as part of the input data.

The results of inference will be written into the specified output_file, with each row corresponding to a 2 label probability prediction of the inputs (first index (0) is probability for benign, second index (1) is probability for pathogenic).

##### Example

To run a basic example, download this example zip folder: www.openbioinformatics.org/mutformer/basic_example.zip. Unzip the file, navigate through the Terminal to the unzipped folder, and create a conda environment as normal. Activate the environment, then run:

```
python run_inference.py --input_file="input_file.txt" --output_file="output_file.txt" --model_folder="model_ckpt_folder" --model_architecture="MutFormer_embedded_convs" --vocab_file="vocab.txt" --batch_size=8 --max_seq_length=1024 --use_ex_data=False
```

The inference results of MutFormer on the input file will be in "output_file.txt" in that folder.

## Input Data format guidelines:

### General format:

Each residue in each sequence should be separated by a space, and to denote the actual start and finish of each entire sequence, a "B" should be placed at the start of each sequence and a "J" at the end of the sequence prior to trimming/splitting.

for pretraining, datasets should be split into "train.txt", "eval.txt", and "test.txt"
for finetuning, datasets should be split into "train.tsv", "dev.tsv", and "test.tsv"

During finetuning, whenever splitting was required, we placed the mutation at the most center point possible, and the rest was trimmed off. 

### Inference

When compiling data for inference, data should be prepared in the following way (inference uses paired sequence method, as this was the best performing strategy):
    
The format should be a tsv file with each line containing (tab delimited): 
1. reference sequence
2. mutated sequence
3. external data, if desired; specify the option when running the inference scripts, and include external data as float values separated by spaces.

For the trained MutFormer models, external data was included according to the following format: 

* All predictions were normalized to values between 1 and 2 by calculating each individual value, if provided, as 1+((provided_value + minimum value in all of DBNSFP) / (maximum value in all of DBNSFP-minimum value in all of DBNSFP)). All values that were missing were assigned a value of 0.
* The data was gathered from the DBNSFPv3 dataset; all columns used for MutFormer are (27 total): 
    1. SIFT_score,
    2. Polyphen2_HDIV_score,
    3. Polyphen2_HVAR_score,
    4. LRT_score,
    5. MutationTaster_score,
    6. MutationAssessor_score,
    7. FATHMM_score,
    8. PROVEAN_score,
    9. VEST3_score,
    10. CADD_raw,
    11. CADD_phred,
    12. DANN_score,
    13. fathmm-MKL_coding_score,
    14. MetaSVM_score,
    15. MetaLR_score,
    16. integrated_fitCons_score,
    17. GERP++_RS,
    18. phyloP7way_vertebrate,
    19. phyloP20way_mammalian,
    20. phastCons7way_vertebrate,
    21. phastCons20way_mammalian,
    22. SiPhy_29way_logOdds,
    23. VARITY_R,
    24. VARITY_ER,
    25. VARITY_R_LOO,
    26. VARITY_ER_LOO,
    27. MVP_score

Example file (with external data):
```
D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F    1.6 1.137 1.5 1.9812
S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L    1.0 1.101 1.1 1.0001
L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T    2.0 1.986 1.8 1.9995
L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L    1.0 1.012 1.0 1.0
```
    
Example file (without external data):
```
D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F
S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L
L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T
L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L
```
    
## For Reproducible Workflow

For a reproducible workflow of pretraining and finetuning the MutFormer models, see Reproducible_Workflow.txt.

#### Note:

MutFomer's model code was written in Tensorflow, and training and inference were run using the TPUEstimator API for Tensorflow 1.15 on cloud TPUs provided by either Google Colab or Google Cloud Platform. For this reason, the notebooks used to both train and finetune MutFormer are built for usage in Google Colab on cloud TPUs. 

Because Colab TPUs can only communicate with cloud storage buckets and not google drive, in order to run Mutformer on cloud TPUs using Colab, one should create a storage bucket through Google Cloud Storage (https://cloud.google.com/storage) and paste the name of the bucket in the "BUCKET_NAME" field in the "Configure settings" of each Colab notebook (GCS provides $300 free credit to new users).

Google Colab has officially removed support for Tensorflow 1.15 as of late 2022; the notebooks were updated and tested to function properly as of late Janurary 2023, but if version errors arise when attempting to run a notebook, please raise an issue with the notebook settings as well as the error message and corresponding code block.

# Citation

If you use MutFormer, please cite the [arXiv paper](https://arxiv.org/abs/2110.14746v1): 

> Jiang, T., Fang, L. & Wang, K. MutFormer: A context-dependent transformer-based model to predict pathogenic missense mutations. Preprint at https://arxiv.org/abs/2110.14746 (2022).

Bibtex format: 
```
@article{jiang2021mutformer,
    title={MutFormer: A context-dependent transformer-based model to predict pathogenic missense mutations}, 
    author={Theodore Jiang and Li Fang and Kai Wang},
    journal={arXiv preprint arXiv:2110.14746},
    year={2022}
}
```
