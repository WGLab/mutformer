from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow import keras as keras
import sys
sys.path.append("C:/Users/JiangQin/Documents/python/transformer project updated/code/updated-working/mutformer_model_code")
import modeling as modeling
import run_classifier as run_classifier

init_checkpoint = "C:/Users/JiangQin/Documents/python/transformer_project_updated/models/pathgenicity prediction"
init_checkpoint = tf.train.latest_checkpoint(init_checkpoint)

#test = "[CLS] B M V T E F I F L G L S D S Q E L Q T F L F M L F F V F Y G G I V F G N L L I V I T V V S D S H L H S P M Y F L L A N L S L I D L S L S S V T A P K M I T D F F S Q R K V I S F K G C L V Q I F L L H F F G G S E M V I L I A M G F D R Y I A I C K P L H Y T T I M C G N A C V G I M A V T W G I G F L H S V S Q L A F A V H L L F C G P N E V D S F Y C D L P R V I K L A C T D T Y R L D I M V I A N S G V L T V C S F V L L I I S Y T I I L M T I Q H R P L D K S S K A L S T L T A H I T V V L L F F G P C V F I Y A W [SEP] B M A T E F I F L G L S D S Q E L Q T F L F M L F F V F Y G G I V F G N L L I V I T V V S D S H L H S P M Y F L L A N L S L I D L S L S S V T A P K M I T D F F S Q R K V I S F K G C L V Q I F L L H F F G G S E M V I L I A M G F D R Y I A I C K P L H Y T T I M C G N A C V G I M A V T W G I G F L H S V S Q L A F A V H L L F C G P N E V D S F Y C D L P R V I K L A C T D T Y R L D I M V I A N S G V L T V C S F V L L I I S Y T I I L M T I Q H R P L D K S S K A L S T L T A H I T V V L L F F G P C V F I Y A W [SEP] [PAD]"
#test = "[CLS] M L G S L A A L A A L A V I G D R P S L T H V V E W I D F E T L A L L F G M M I L V A I F S E T G F F D Y C A V K A Y R L S R G R V W A M I I M L C L I A A V L S A F L D N V T T M L L F T P V T I R L C E V L N L D P R Q V L I A E V I F T N I G G A A T A I G D P P N V I I V S N Q E L R K M G L D F A G F T A H M F I G I C L V L L V C F P L L R L L Y W N R K L Y N K E P S E I V E L K H E I H V W R L T A Q R I S P A S R E E T A V R R L L L G K V L A L E H L L A R R L H T F H R Q I S Q E D K N W E T N I Q E [SEP] M L G S L A A L A A L A V I G D R P S L T H V V E W I D F E T L A L L F G M M I L V A I F S E T G F F D Y C A V K A Y R L S R G R V W A M I I M L C L I A A V L S A F L D N V T T M L L F T P V T I R L C E V L N L D P R Q V L I A E V I F T N I G G A A T A I V D P P N V I I V S N Q E L R K M G L D F A G F T A H M F I G I C L V L L V C F P L L R L L Y W N R K L Y N K E P S E I V E L K H E I H V W R L T A Q R I S P A S R E E T A V R R L L L G K V L A L E H L L A R R L H T F H R Q I S Q E D K N W E T N I Q E [SEP] [PAD]"
#test = "[CLS] T E Q A E A P C R G Q A C S A Q K A Q P V G T C P G E E W M I R K V K V E D E D Q E A E E E V E W P Q H L S L L P S P F P A P D L G H L A A A Y K L E P G A P G A L S G L A L S G W G P M P E K P Y G C G E C E R R F R D Q L T L R L H Q R L H R G E G P C A C P D C G R S F T Q R A H M L L H Q R S H R G E R P F P C S E C D K R F S K K A H L T R H L R T H T G E R P Y P C A E C G K R F S Q K I H L G S H Q K T H T G E R P F P C T E C E K R F R K K T H L I R H Q R I H T G E R P Y Q C A Q C A R S F T H K Q H L V [SEP] T E Q A E A P C R G Q A C S A Q K A Q P V G T C P G E E W M I R K V K V E D E D Q E A E E E V E W P Q H L S L L P S P F P A P D L G H L A A A Y K L E P G A P G A L S G L A L S G W G P M P E K P Y G C G E C E R R F R D Q L T L R L H Q R L H R G E G P C A C R D C G R S F T Q R A H M L L H Q R S H R G E R P F P C S E C D K R F S K K A H L T R H L R T H T G E R P Y P C A E C G K R F S Q K I H L G S H Q K T H T G E R P F P C T E C E K R F R K K T H L I R H Q R I H T G E R P Y Q C A Q C A R S F T H K Q H L V [SEP] [PAD]"
test = "[CLS] A S E S R C Q Q G K T Q F G V G L R S G G E N H L W L L E G T P S L Q S C W A A C C Q D S A C H V F W W L E G M C I Q A D C S R P Q S C R A F R T H S S N S M L V F L K K F Q T A D D L G F L P E D D V P H L L G L G W N W A S W R Q S P P R A A L R P A V S S S D Q Q S L I R K L Q K R G S P S D V V T P I V T Q H S K V N D S N E L G G L T T S G S A E V H K A I T I S S P L T T D L T A E L S G G P K N V S V Q P E I S E G L A T T P S T Q Q V K S S E K T Q I A V P Q P V A P S Y S Y A T P T P Q A S F Q S T S A P [SEP] A S E S R C Q Q G K T Q F G V G L R S G G E N H L W L L E G T P S L Q S C W A A C C Q D S A C H V F W W L E G M C I Q A D C S R P Q S C R A F R T H S S N S M L V F L K K F Q T A D D L G F L P E D D V P H L L G L G W N W A S W R Q S P P R A A L R P A V S S C D Q Q S L I R K L Q K R G S P S D V V T P I V T Q H S K V N D S N E L G G L T T S G S A E V H K A I T I S S P L T T D L T A E L S G G P K N V S V Q P E I S E G L A T T P S T Q Q V K S S E K T Q I A V P Q P V A P S Y S Y A T P T P Q A S F Q S T S A P [SEP] [PAD]"

file = "C:/Users/JiangQin/Documents/python/transformer_project_updated/data/final_finetuning_data/BERT finetuning no redundancies/MRPC/modified_bert_mrpc_512/test.tsv"
#file = "C:/Users/JiangQin/Documents/python/transformer_project_updated/data/full_database_prediction/test_tiny.txt"
vocab = \
'''[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
L
S
B
J
E
A
P
T
G
V
K
R
D
Q
I
N
F
H
Y
C
M
W'''

tokenss = []
segment_idss = []
input_maskss = []
labels = []

for line in open(file).read().split("\n"):
    tokens = ["[CLS]"]
    segment_ids = [0]

    stuff = line.split("\t")

    labels.append(stuff[0])

    for char in stuff[3].split(" "):
        tokens.append(char)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for char in stuff[4].split(" "):
        tokens.append(char)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_mask = [1 for x in tokens]
    while len(tokens) < 512:
        tokens.append("[PAD]")
        segment_ids.append(0)
        input_mask.append(0)
    tokens = [vocab.split("\n").index(x) for x in tokens]
    tokenss.append(tokens)
    segment_idss.append(segment_ids)
    input_maskss.append(input_mask)

config = {
  "hidden_size": 768,
  "hidden_act": "gelu",
  "initializer_range": 0.02,
  "hidden_dropout_prob": 0.1,
  "num_attention_heads": 12,
  "type_vocab_size": 2,
  "max_position_embeddings": 1024,
  "num_hidden_layers": 12,
  "intermediate_size": 3072,
  "attention_probs_dropout_prob": 0.1,
  "vocab_size": 27
}

num_labels = 2

config = modeling.BertConfig.from_dict(config)

amt = 100

with open("asdf.txt", "w+") as out:
    for i in range(0,(len(tokenss)//100)+1):
        print(i)
        print("\n\n\n\n\n\n\n\n\n\n\n\n"+str(i/(len(tokenss)//100))+"\n\n\n\n\n\n\n\n\n\n\n\n\n")
        tokens = tf.constant(tokenss[amt*i:amt*(i+1)])
        masks = tf.constant(input_maskss[amt*i:amt*(i+1)])
        segments = tf.constant(segment_idss[amt*i:amt*(i+1)])

        print(tokens,masks,segments)


        model = modeling.BertModelModified(
            config=config,
            is_training=False,
            input_ids=tokens,
            input_mask=masks,
            token_type_ids=segments,
            use_one_hot_embeddings=True)


        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)



        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                           modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            in_chkpt = [var for var in tvars if var.name in initialized_variable_names]
            not_in_chkpt = [var for var in tvars if var.name not in initialized_variable_names]
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        init = tf.global_variables_initializer()
        print(not_in_chkpt)
        print(init_checkpoint)

        with tf.Session() as sess:
            sess.run(init)
            #saver = tf.train.import_meta_graph("C:/Users/JiangQin/Documents/python/transformer_project_updated/models/pathgenicity prediction/model.ckpt-10000.meta")
            #saver.restore(sess, init_checkpoint)
            asdf = sess.run(probabilities)
            out.write("\n".join([str(x) for x in asdf])+"\n")
        tf.compat.v1.reset_default_graph()