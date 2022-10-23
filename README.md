# MutFormer
MutFormer is an application of the BERT (Bidirectional Encoder Representations from Transformers) NLP (Natural Language Processing) model with an added adaptive vocabulary to protein context, for the purpose of predicting the effect of missense mutations on protein function.

For this project, a total of 5 models were trained:

Model Name | Hidden Layers | Hidden Size (and size of convolution filters) | Intermediate Size | Input length | # of parameters | Download link
-----------|---------------|-------------|-------------------|--------------|-----------------|--------------
MutBERT8L | 8 | 768 | 3072 | 1024 | ~58M | https://drive.google.com/drive/folders/1dJwSPWOU8VVLwQbe8UlxSLyAiJqCWszn?usp=sharing
MutBERT10L | 10 | 770 | 3072 | 1024 | ~72M | https://drive.google.com/drive/folders/1--nJNAwCB5weLH8NclNYJsrYDx2DZUhj?usp=sharing
MutFormer8L | 8 | 768 | 3072 | 1024 | ~62M | https://drive.google.com/drive/folders/1-LXP5dpO071JYvbxRaG7hD9vbcp0aWmf?usp=sharing
MutFormer10L | 10 | 770 | 3072 | 1024 | ~76M | https://drive.google.com/drive/folders/1-GWOe1uiosBxy5Y5T_3NkDbSrv9CXCwR?usp=sharing
MutFormer12L (Same size transformer as BERT-base) | 12 | 768 | 3072 | 1024 | ~86M | https://drive.google.com/drive/folders/1-59X7Wu7OMDB8ddnghT5wvthbmJ9vjo5?usp=sharing
MutFormer8L (integrated adap vocab) | 8 | 768 | 3072 | 1024 | ~64M | https://drive.google.com/drive/folders/1jcK2mckj_oJaR1QQVzjBuoOEJC5SWDu5?usp=sharing

MutBERT8L and MutBERT10L use the original BERT model for comparison purposes, the MutFormer models are the official models.


#### To download a full prediction of a complete collection of all possible known missense protein-altering mutations in the humane proteome, we have included a file as a release asset called "hg19_mutformer.zip" 

#### Alternatively, a google drive link: https://drive.google.com/file/d/1950d_f3y9Q6C5I62ODjHB6C8biT8whY7/view?usp=sharing


## To run MutFormer:

MutFomer's model code was written in Tensorflow, and training and inference were run using the TPUEstimator API on cloud TPUs provided by either Google Colab or Google Cloud Platform. For this reason, the notebooks used to both train and finetune MutFormer are built for usage in Google Colab on cloud TPUs. To perform inference using MutFormer, see the below "Inference" section which will document usage of code in the "mutformer_inference" folder, will provide both colab/cloud TPU and local machine support.

### Pretraining:

Under the folder titled "mutformer_pretraining," first open "mutformer_pretraining_data generation_(with dynamic masking op).ipynb," and run through the code segments (if using colab, runtime options: Hardware Accelerator-None, Runtime shape-Standard), selecting the desired options along the way, to generate eval and test data, as well as begin the constant training data generation with dynamic masking.

Once the data generation has begun, open "mutformer_run_pretraining.ipynb," and in a different runtime, run the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-High RAM if available, Standard otherwise) to start the training.

Finally, open "mutformer_run_pretraining_eval.ipynb" and run all the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-Standard) in another runtime to begin the parallel evaluation operation.


You can make multiple copies of the data generation and run_pretraining scripts to train multiple models at a time. The evaluation script is able to handle evaluating multiple models at once.

To view pretraining graphs or download the checkpoints from GCS, use the notebook titled “mutformer_processing_and_viewing_pretraining_results.”

### Finetuning

For finetuning, there is only one set of files for four modes, so at the top of each notebook there is an option to select the desired mode to use (MRPC for paired strategy, MRPC_w_preds for MRPC with external predictions, RE for single sequence strategy, and NER for pre residue strategy).

Under the folder titled "mutformer_finetuning," first open "mutformer_finetuning_data_generation.ipynb," and run through the code segments (if using colab, runtime options: Hardware Accelerator-None, Runtime shape-Standard), selecting the desired options along the way, to generate train,eval,and test data.

Once the data generation has finished, open "mutformer_finetuning_benchmark.ipynb," and in a different runtime, run the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-High RAM if available, Standard otherwise). There are three different options to use: either training multiple models on different sequence lengths, training just one model on multiple sequence lengths with different batch sizes, or training just one single model with specified sequence lengths and specified batch sizes. There are also options for whether to run prediction or evaluation, and which dataset to use.

Finally, alongside running mutformer_run_finetuning, open "mutformer_finetuning_benchmark_eval_predict.ipynb" and run all the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-Standard) in another runtime to begin the parallel evaluation operation (can also evaluate or predict after the fact).

### Inference

To run the trained MutFormer model, one can choose either to run MutFormer on a cloud TPU via Colab, for which a Colab notebook is provided, or run inference locally, for which a python script is provided.

#### Local Run

To run MutFormer for inference locally, download MutFormer code and navigate via Terminal into the folder:  "mutformer/mutformer_inference/local_run." Run the script "run_inference.py," specifying the parameters.

Parameters for run_inference.py include:

--input_file: Input raw text file for running inference (or comma-separated list of files) (should be formatted corresponding to the test mode chosen from the Finetuning section in the README).

--output_file: File to output inference results to (or comma-separated list of files).

--model_folder: Folder where the model checkpoint and config file is stored.

--model_architecture: Model architecture of the model checkpoint specified (BertModel indicates the original BERT, BertModelModified indicates MutFormer's architecture without integrated convs, MutFormer_embedded_convs indicates MutFormer with integrated convolutions).

--vocab_file: The vocabulary file that the BERT model was trained on.

--batch_size: Number of lines to batch together at a time when performing inference.

--max_seq_length: Maximum sequence length to pad to; should be greater than or equal to the three more than twice the length of the longest input amino acid sequence.

--using_ex_data: set this to True if external data is being included as part of the input data.


##### Example

To run a basic example, download the "basic_example.zip" from "mutformer_inference/local_run," navigate through the Terminal to the unzipped folder, and run the following command:

```
python run_inference.py --input_file="input_file.txt" --output_file="output_file.txt" --model_folder="model_ckpt_folder" --model_architecture="MutFormer_embedded_convs" --vocab_file="vocab.txt" --batch_size=64 
```

The inference results of MutFormer on the input file will be in "output_file.txt" in that folder.

<Cloud Run>

<UNFINISHED>

## Input Data format guidelines:

### General format:

Each residue in each sequence should be separated by a space, and to denote the actual start and finish of each entire sequence, a "B" should be placed at the start of each sequence and a "J" at the end of the sequence prior to trimming/splitting.

for pretraining, datasets should be split into "train.txt", "eval.txt", and "test.txt"
for finetuning, datasets should be split into "train.tsv", "dev.tsv", and "test.tsv"

During finetuning, whenever splitting was required, we placed the mutation at the most center point possible, and the rest was trimmed off. 

### Pretraining:

#### We have included our pretraining data in this repository as an asset, called "pretraining_data.zip" 

#### Alternatively, a google drive link: https://drive.google.com/drive/folders/1QlTx0iOS8aVKnD0fegkG5JOY6WGH9u_S?usp=sharing

The format should be a txt with each line containing one sequence. Each sequence should be trimmed/split to a maximum of a fixed length (in our case we used 1024 amino acids). 

Example file:
```
B M E T A V I G V V V V L F V V T V A I T C V L C C F S C D S R A Q D P Q G G P G J
B M V S S Y L V H H G Y C A T A T A F A R M T E T P I Q E E Q A S I K N R Q K I Q K 
L V L E G R V G E A I E T T Q R F Y P G L L E H N P N L L F M L K C R Q F V E M V N 
G T D S E V R S L S S R S P K S Q D S Y P G S P S L S F A R V D D Y L H J
```

### Finetuning training (not inference)
#### Single Sequence Classification (RE)

The format should be a tsv file with each line containing (tab delimited): 
1.  mutated protein sequence
2.  label (1 for pathogenic and 0 for benign). 

Example file:
```
V R K T T S P E G E V V P L H Q V D I P M E N G V G G N S I F L V A P L I I Y H V I D A N S P L Y D L A P S D L H H H Q D L    0
P S I P T D I S T L P T R T H I I S S S P S I Q S T E T S S L V V T T S P T M S T V R M T L R I T E N T P I S S F S T S I V    0
G Q F L L P L T Q E A C C V G L E A G I N P T D H L I T A Y R A Q G F T F T R G L S V R E I L A E L T G R K G G C A K G K G    1
P A G L G S A R E T Q A Q A C P Q E G T E A H G A R L G P S I E D K G S G D P F G R Q R L K A E E M D T E D R P E A S G V D    0
```

#### Per Residue Classification (NER)

The format should be a tsv file with each line containing (tab delimited): 
1.  mutated protein sequence
2.  per residue labels
3.  mutation position (index; if the 5th residue is mutated the mutation position would be 4) ("P" for pathogenic and "B" for benign). 

The per residue labels should be the same length as the mutated protein sequence. Every residue is labelled as "B" unless it was a mutation site, in which case it was labelled either "B" or "P." The loss is calculated on only the mutation site.


Example file:
```
F R E F A F I D M P D A A H G I S S Q D G P L S V L K Q A T    B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B    16
A T D L D A E E E V V A G E F G S R S S Q A S R R F G T M S    B B B B B B B B B B B B B B B P B B B B B B B B B B B B B B    16
G K K G D V W R L G L L L L S L S Q G Q E C G E Y P V T I P    B B B B B B B B B B B B B B B P B B B B B B B B B B B B B B    16
E M C Q K L K F F K D T E I A K I K M E A K K K Y E K E L T    B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B    16
```

#### Paired Sequence Classification (MRPC)

The format should be a tsv file with each line containing (tab delimited): 
1. label (1 for pathogenic and 0 for benign)
2. reference sequence
3. mutated sequence

Example file:
```
1    D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F
0    S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L
1    L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T
0    L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L
```

#### Paired Sequence Classification With External Data (MRPC_w_ex_data)
    
The format should be a tsv file with each line containing (tab delimited): 
1. label (1 for pathogenic and 0 for benign)
2. reference sequence
3. mutated sequence
4. external data (float values separated by spaces)

Example file:
```
1    D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F    0.6 0.137 0.5 0.9812
0    S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L    0.0 0.101 0.1 0.0001
1    L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T    1.0 0.986 0.8 0.9995
0    L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L    0.0 0.012 0.0 0.0
```

### Inference

When compiling data for inference, data should be prepared in the following way:
    
The format should be a tsv file with each line containing (tab delimited): 
1. reference sequence
2. mutated sequence
3. If using external data, specify the option when running the inference scripts, and include external data (float values separated by spaces) in the data.
    For the trained MutFormer models, external data was included according to the DBNSFP database's ordering of external predictions.

Example file (with external data):
```
D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F    0.6 0.137 0.5 0.9812
S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L    0.0 0.101 0.1 0.0001
L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T    1.0 0.986 0.8 0.9995
L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L    0.0 0.012 0.0 0.0
```
    
Example file (without external data):
```
D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F
S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L
L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T
L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L
```
    
# Citation

If you use MutFormer, please cite the [arXiv paper](https://arxiv.org/abs/2110.14746v1): 

> Jiang, T., Fang, L. & Wang, K. MutFormer: A context-dependent transformer-based model to predict pathogenic missense mutations. Preprint at https://arxiv.org/abs/2110.14746 (2021).

Bibtex format: 
```
@article{jiang2021mutformer,
    title={MutFormer: A context-dependent transformer-based model to predict pathogenic missense mutations}, 
    author={Theodore Jiang and Li Fang and Kai Wang},
    journal={arXiv preprint arXiv:2110.14746},
    year={2021}
}
```



