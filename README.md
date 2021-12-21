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

Orig BERT small and Orig BERT medium use the original BERT model for comparison purposes, the MutFormer models the official models.

Best performing MutFormer model for functional effect prediction:

https://drive.google.com/drive/folders/1tsC0lqzbx3wR_jOer9GuGjeJnnYL4RND?usp=sharing


#### To download a full prediction of a mostly complete collection of all possible missense proteins in the humane proteome, we have included a file as an asset called "hg19_mutformer.zip" 

#### Alternatively, a google drive link: https://drive.google.com/file/d/1ObBEn-wcQwoebD7glx8bWiWILfzfnlIO/view?usp=sharing


## To run MutFormer:

### Pretraining:

Under the folder titled "MutFormer_pretraining," first open "MutFormer_pretraining_data generation_(with dynamic masking op).ipynb," and run through the code segments (if using colab, runtime options: Hardware Accelerator-None, Runtime shape-Standard), selecting the desired options along the way, to generate eval and test data, as well as begin the constant training data generation with dynamic masking.

Once the data generation has begun, open "MutFormer_run_pretraining.ipynb," and in a different runtime, run the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-High RAM if available, Standard otherwise) to start the training.

Finally, open "MutFormer_run_pretraining_eval.ipynb" and run all the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-Standard) in another runtime to begin the parallel evaluation operation.


You can make multiple copies of the data generation and run_pretraining scripts to train multiple models at a time. The evaluation script is able to handle evaluating multiple models at once.

To view pretraining graphs or download the checkpoints from GCS, use the notebook titled “MutFormer_processing_and_viewing_pretraining_results.”

### Finetuning

For finetuning, there is only one set of files for three modes, so at the top of each notebook there is an option to select the desired mode to use (MRPC for paired strategy, RE for single sequence strategy, and NER for pre residue strategy).

Under the folder titled "MutFormer_finetraining," first open "MutFormer_finetuning_data_generation.ipynb," and run through the code segments (if using colab, runtime options: Hardware Accelerator-None, Runtime shape-Standard), selecting the desired options along the way, to generate train,eval,and test data.

Once the data generation has finished, open "MutFormer_finetuning_benchmark.ipynb," and in a different runtime, run the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-High RAM if available, Standard otherwise). There are three different options to use: either training multiple models on different sequence lengths, training just one model on multiple sequence lengths with different batch sizes, or training just one single model with specified sequence lengths and specified batch sizes. There are also options for whether to run prediction or evaluation, and which dataset to use.

Finally, alongside running MutFormer_run_finetuning "MutFormer_finetuning_benchmark_eval.ipynb" and run all the code segments there (if using colab, runtime options: Hardware Accelerator-TPU, Runtime shape-Standard) in another runtime to begin the parallel evaluation operation.

To view finetuning graphs or plot ROC curves for the predictions, use the notebook titled “MutFormer_processing_and_viewing_finetuning_pathogenic_variant_classification_(2_class)_results.ipynb.”

## Model top performances for Pathogenicity Prediction:

Model Name | Receiver Operator Characteristic Area Under Curve (ROC AUC) 
-----------|---------------
MutBERT8L | 0.845
MutBERT10L | 0.876
MutFormer8L | 0.931
MutFormer10L | 0.932
MutFormer12L | 0.933

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

### Finetuning
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
1.  label (1 for pathogenic and 0 for benign)
3.  reference sequence
4.  mutated sequence

Example file:
```
1    asdf    D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F
0    asdf    S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L
1    asdf    L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T
0    asdf    L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L
```

#### Paired Sequence Classification With External Data (MRPC_w_ex_data)

The format should be a tsv file with each line containing (tab delimited): 
1.  label (1 for pathogenic and 0 for benign)
3.  reference sequence
4.  mutated sequence
5.  predictions, separated by spaces

Example file:
```
1    D W A Y A A S K E S H A T L V F H N L L G E I D Q Q Y S R F    D W A Y A A S K E S H A T L V F Y N L L G E I D Q Q Y S R F    0.6 0.137 0.5 0.9812
0    S A V P P F S C G V I S T L R S R E E G A V D K S Y C T L L    S A V P P F S C G V I S T L R S W E E G A V D K S Y C T L L    0.0 0.101 0.1 0.0001
1    L L D S S L D P E P T Q S K L V R L E P L T E A E A S E A T    L L D S S L D P E P T Q S K L V H L E P L T E A E A S E A T    1.0 0.986 0.8 0.9995
0    L A E D E A F Q R R R L E E Q A A Q H K A D I E E R L A Q L    L A E D E A F Q R R R L E E Q A T Q H K A D I E E R L A Q L    0.0 0.012 0.0 0.0
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



