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
#test = "[CLS] A S E S R C Q Q G K T Q F G V G L R S G G E N H L W L L E G T P S L Q S C W A A C C Q D S A C H V F W W L E G M C I Q A D C S R P Q S C R A F R T H S S N S M L V F L K K F Q T A D D L G F L P E D D V P H L L G L G W N W A S W R Q S P P R A A L R P A V S S S D Q Q S L I R K L Q K R G S P S D V V T P I V T Q H S K V N D S N E L G G L T T S G S A E V H K A I T I S S P L T T D L T A E L S G G P K N V S V Q P E I S E G L A T T P S T Q Q V K S S E K T Q I A V P Q P V A P S Y S Y A T P T P Q A S F Q S T S A P [SEP] A S E S R C Q Q G K T Q F G V G L R S G G E N H L W L L E G T P S L Q S C W A A C C Q D S A C H V F W W L E G M C I Q A D C S R P Q S C R A F R T H S S N S M L V F L K K F Q T A D D L G F L P E D D V P H L L G L G W N W A S W R Q S P P R A A L R P A V S S C D Q Q S L I R K L Q K R G S P S D V V T P I V T Q H S K V N D S N E L G G L T T S G S A E V H K A I T I S S P L T T D L T A E L S G G P K N V S V Q P E I S E G L A T T P S T Q Q V K S S E K T Q I A V P Q P V A P S Y S Y A T P T P Q A S F Q S T S A P [SEP]"
#test = "Y S S W D A M C Y L D P S K A V E E D D F V V G F W N P S E E N C G V D T G K Q S I S Y D L H T E Q C I A D K S I A D C V E A L L G C Y L T S C G E R A A Q L F L C S L G L K V L P V I K R T D R E K A L C P T R E N F N S Q Q K N L S V S C A A A S V A S S R S S V L K D S E Y G C L K I P P R C M F D H P D A D K T L N H L I S G F E N F E K K I N Y R F K N K A Y L L Q A F T H A S Y H Y N T I T D C Y Q R L E F L G D A I L D Y L I T K H L Y E D P R Q H S P G V L T D L R S A L V N N T I F A S L A V K Y D Y H K Y S S W D A M C Y L D P S K A V E E D D F V V G F W N P S E E N C G V D T G K Q S I S Y D L H T E Q C I A D K S I A D C V E A L L G C Y L T S C G E R A A Q L F L C S L G L K V L P V I K R T D R E K A L C P T R E N F N S Q Q K N L S V S C A A A S V A S S R A S V L K D S E Y G C L K I P P R C M F D H P D A D K T L N H L I S G F E N F E K K I N Y R F K N K A Y L L Q A F T H A S Y H Y N T I T D C Y Q R L E F L G D A I L D Y L I T K H L Y E D P R Q H S P G V L T D L R S A L V N N T I F A S L A V K Y D Y H K"
#test = "M L G S L A A L A A L A V I G D R P S L T H V V E W I D F E T L A L L F G M M I L V A I F S E T G F F D Y C A V K A Y R L S R G R V W A M I I M L C L I A A V L S A F L D N V T T M L L F T P V T I R L C E V L N L D P R Q V L I A E V I F T N I G G A A T A I G D P P N V I I V S N Q E L R K M G L D F A G F T A H M F I G I C L V L L V C F P L L R L L Y W N R K L Y N K E P S E I V E L K H E I H V W R L T A Q R I S P A S R E E T A V R R L L L G K V L A L E H L L A R R L H T F H R Q I S Q E D K N W E T N I Q E	M L G S L A A L A A L A V I G D R P S L T H V V E W I D F E T L A L L F G M M I L V A I F S E T G F F D Y C A V K A Y R L S R G R V W A M I I M L C L I A A V L S A F L D N V T T M L L F T P V T I R L C E V L N L D P R Q V L I A E V I F T N I G G A A T A I V D P P N V I I V S N Q E L R K M G L D F A G F T A H M F I G I C L V L L V C F P L L R L L Y W N R K L Y N K E P S E I V E L K H E I H V W R L T A Q R I S P A S R E E T A V R R L L L G K V L A L E H L L A R R L H T F H R Q I S Q E D K N W E T N I Q E"
#test = "E Q R L N R H L A E V L E R V N S K G Y K V Y G A G S S L Y G G T I T I N A R K F E E M N A E L E E N K E L A Q N R L C E L E K L R Q D F E E V T T Q N E K L K V E L R S A V E Q V V K E T P E Y R C M Q S Q F S V L Y N E S L Q L K A H L D E A R T L L H G T R G T H Q H Q V E L I E R D E V S L H K K L R T E V I Q L E D T L A Q V R K E Y E M L R I E F E Q T L A A N E Q A G P I N R E M R H L I S S L Q N H N H Q L K G E V L R Y K R K L R E A Q S D L N K T R L R S G S A L L Q S Q S S T E D P K D E P A E L K P	E Q R L N R H L A E V L E R V N S K G Y K V Y G A G S S L Y G G T I T I N A R K F E E M N A E L E E N K E L A Q N R L C E L E K L R Q D F E E V T T Q N E K L K V E L R S A V E Q V V K E T P E Y R C M Q S Q F S V L Y N E S L Q L K A H L D E A R T L L H G T K G T H Q H Q V E L I E R D E V S L H K K L R T E V I Q L E D T L A Q V R K E Y E M L R I E F E Q T L A A N E Q A G P I N R E M R H L I S S L Q N H N H Q L K G E V L R Y K R K L R E A Q S D L N K T R L R S G S A L L Q S Q S S T E D P K D E P A E L K P"
test = "T E Q A E A P C R G Q A C S A Q K A Q P V G T C P G E E W M I R K V K V E D E D Q E A E E E V E W P Q H L S L L P S P F P A P D L G H L A A A Y K L E P G A P G A L S G L A L S G W G P M P E K P Y G C G E C E R R F R D Q L T L R L H Q R L H R G E G P C A C P D C G R S F T Q R A H M L L H Q R S H R G E R P F P C S E C D K R F S K K A H L T R H L R T H T G E R P Y P C A E C G K R F S Q K I H L G S H Q K T H T G E R P F P C T E C E K R F R K K T H L I R H Q R I H T G E R P Y Q C A Q C A R S F T H K Q H L V	T E Q A E A P C R G Q A C S A Q K A Q P V G T C P G E E W M I R K V K V E D E D Q E A E E E V E W P Q H L S L L P S P F P A P D L G H L A A A Y K L E P G A P G A L S G L A L S G W G P M P E K P Y G C G E C E R R F R D Q L T L R L H Q R L H R G E G P C A C R D C G R S F T Q R A H M L L H Q R S H R G E R P F P C S E C D K R F S K K A H L T R H L R T H T G E R P Y P C A E C G K R F S Q K I H L G S H Q K T H T G E R P F P C T E C E K R F R K K T H L I R H Q R I H T G E R P Y Q C A Q C A R S F T H K Q H L V"
test = "B M K T L P L F V C I C A L S A C F S F S E G R E R D H E L R H R R H H H Q S P K S H F E L P H Y P G L L A H Q K P F I R K S Y K C L H K R C R P K L P P S P N N P P K F P N P H Q P P K H P D K N S S V V N P T L V A T T Q I P S V T F P S A S T K I T T L P N V T F L P Q N A T T I S S R E N V N T S S S V A T L A P V N S P A P Q D T T A A P P T P S A T T P A P P S S S A P P E T T A A P P T P S A T T Q A P P S S S A P P E T T A A P P T P P A T T P A P P S S S A P P E T T A A P P T P S A T T P A P L S S S A	B M K T L P L F V C I C A L S A C F S F S E G R E R D H E L R H R R H H H Q S P K S H F E L P H Y P G L L A H Q K P F I R K S Y K C L H K R C R P K L P P S P N N P P K F P N P H Q P P K H P D K N S S V V N P T L V A K T Q I P S V T F P S A S T K I T T L P N V T F L P Q N A T T I S S R E N V N T S S S V A T L A P V N S P A P Q D T T A A P P T P S A T T P A P P S S S A P P E T T A A P P T P S A T T Q A P P S S S A P P E T T A A P P T P P A T T P A P P S S S A P P E T T A A P P T P S A T T P A P L S S S A"
test = "[CLS] "+" [SEP] ".join(test.split("\t"))+" [SEP]"
print(test)





#file = "C:/Users/JiangQin/Documents/python/transformer_project_updated/data/final_finetuning_data/BERT finetuning no redundancies/MRPC/modified_bert_mrpc_512/test.tsv"
file = "C:/Users/JiangQin/Documents/python/transformer_project_updated/data/full_database_prediction/test_tiny.txt"
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


tokens = test.split(" ")

segment_ids = []
index = 0
for token in tokens:
    segment_ids.append(index)
    if token=="[SEP]":
        index+=1

tokens = [vocab.split("\n").index(x) for x in tokens]
input_mask = [1 for x in tokens]

tokens.append(0)
segment_ids.append(0)
input_mask.append(0)

print(tokens)
print(segment_ids)
print(input_mask)

tokens = tf.constant([tokens])
segment_ids = tf.constant([segment_ids])
input_mask = tf.constant([input_mask])



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


model = modeling.BertModelModified(
    config=config,
    is_training=False,
    input_ids=tokens,
    input_mask=input_mask,
    token_type_ids=segment_ids,
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
    print(asdf)
tf.compat.v1.reset_default_graph()