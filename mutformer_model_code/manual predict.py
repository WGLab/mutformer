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

test = "[CLS] B M V T E F I F L G L S D S Q E L Q T F L F M L F F V F Y G G I V F G N L L I V I T V V S D S H L H S P M Y F L L A N L S L I D L S L S S V T A P K M I T D F F S Q R K V I S F K G C L V Q I F L L H F F G G S E M V I L I A M G F D R Y I A I C K P L H Y T T I M C G N A C V G I M A V T W G I G F L H S V S Q L A F A V H L L F C G P N E V D S F Y C D L P R V I K L A C T D T Y R L D I M V I A N S G V L T V C S F V L L I I S Y T I I L M T I Q H R P L D K S S K A L S T L T A H I T V V L L F F G P C V F I Y A W [SEP] B M M T E F I F L G L S D S Q E L Q T F L F M L F F V F Y G G I V F G N L L I V I T V V S D S H L H S P M Y F L L A N L S L I D L S L S S V T A P K M I T D F F S Q R K V I S F K G C L V Q I F L L H F F G G S E M V I L I A M G F D R Y I A I C K P L H Y T T I M C G N A C V G I M A V T W G I G F L H S V S Q L A F A V H L L F C G P N E V D S F Y C D L P R V I K L A C T D T Y R L D I M V I A N S G V L T V C S F V L L I I S Y T I I L M T I Q H R P L D K S S K A L S T L T A H I T V V L L F F G P C V F I Y A W [SEP]"


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

test = [vocab.index(x) for x in test.split(" ")]
input_mask = [1 for x in test]

segment_ids = []
index = 0
for tes in test:
    segment_ids.append(index)
    if tes==vocab.index("[SEP]"):
       index+=1

print(segment_ids)

config = {
  "hidden_size": 768,
  "hidden_act": "gelu",
  "initializer_range": 0.02,
  "hidden_dropout_prob": 0.1,
  "num_attention_heads": 8,
  "type_vocab_size": 2,
  "max_position_embeddings": 1024,
  "num_hidden_layers": 8,
  "intermediate_size": 3072,
  "attention_probs_dropout_prob": 0.1,
  "vocab_size": 27
}

num_labels = 2

config = modeling.BertConfig.from_dict(config)

print(tf.constant([test]))

model = modeling.BertModelModified(
    config=config,
    is_training=False,
    input_ids=tf.constant([test]),
    input_mask=tf.constant([input_mask]),
    token_type_ids=tf.constant([segment_ids]),
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

init = tf.global_variables_initializer()

tvars = tf.trainable_variables()

initialized_variable_names = {}
scaffold_fn = None
if init_checkpoint:
    (assignment_map, initialized_variable_names) = \
                   modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    in_chkpt = [var for var in tvars if var.name in initialized_variable_names]
    not_in_chkpt = [var for var in tvars if var.name not in initialized_variable_names]
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

print(not_in_chkpt)
print(init_checkpoint)

with tf.Session() as sess:
    #sess.run(init)
    saver = tf.train.import_meta_graph("C:/Users/JiangQin/Documents/python/transformer_project_updated/models/pathgenicity prediction/model.ckpt-10000.meta")
    saver.restore(sess, init_checkpoint)
    print(sess.run(probabilities))

