# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import metric_functions
from tqdm import tqdm
import random

random.seed(31415926525)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, ex_data=None,pos=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.ex_data = ex_data
    self.pos = pos


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               ex_data,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.ex_data = ex_data
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir, read_range=None):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir, read_range=None):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir, read_range=None):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, read_range=None,quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for n,line in enumerate(tqdm(reader,"reading tsv")):
        if read_range:
            if n<read_range[0]:
                continue
            elif n>=read_range[1]:
                break
        lines.append(line)
      return lines



class MrpcProcessor(DataProcessor):
  def get_train_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv"),read_range=read_range), "train")

  def get_dev_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv"),read_range=read_range), "dev")

  def get_test_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv"),read_range=read_range), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(tqdm(lines,"creating_examples")):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      label = tokenization.convert_to_unicode(line[0])
      pos = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos))
    return examples

class MrpcWithExDataProcessor(DataProcessor):
  def get_train_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv"),read_range=read_range), "train")

  def get_dev_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv"),read_range=read_range), "dev")

  def get_test_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv"),read_range=read_range), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(tqdm(lines,"creating_examples")):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      label = tokenization.convert_to_unicode(line[0])
      ex_data = tokenization.convert_to_unicode(line[3])
      pos = tokenization.convert_to_unicode(line[4])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, ex_data=ex_data, pos=pos))
    return examples


class REProcessor(DataProcessor):
  def get_train_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv"),read_range=read_range), "train")

  def get_dev_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv"),read_range=read_range), "dev")

  def get_test_examples(self, data_dir, read_range=None):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv"),read_range=read_range), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(tqdm(lines,"creating_examples")):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      label = tokenization.convert_to_unicode(line[1])
      pos = tokenization.convert_to_unicode(line[2])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label, pos=pos))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer,create_altered_data=False):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  pos = int(example.pos)

  ex_data = example.ex_data
  if ex_data:
    ex_data = [float(ex_dat) for ex_dat in ex_data.split()]

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if create_altered_data:
      def generate_clips(seq,pos):
          start_clip = random.randint(0, int(pos / 2))
          end_clip = random.randint(int((len(seq) + pos) / 2), len(seq))
          return start_clip,end_clip

      start_clip,end_clip = generate_clips(tokens_a,pos)
      tokens_a = tokens_a[start_clip:end_clip + 1]
      if tokens_b:
          tokens_b = tokens_b[start_clip:end_clip + 1]


  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info(f"tokens (length = {len(tokens)}): %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info(f"input_ids (length = {len(input_ids)}): %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info(f"input_mask (length = {len(input_mask)}): %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info(f"segment_ids (length = {len(segment_ids)}): %s" % " ".join([str(x) for x in segment_ids]))
    if ex_data:
        tf.logging.info(f"ex_data (length = {len(ex_data)}): %s" % " ".join([str(x) for x in ex_data]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      ex_data=ex_data,
      is_real_example=True)
  return feature


def shuffle(lst, name):
    newLst = []
    for i in tqdm(range(0, len(lst)), "shuffling " + name):
        ind = random.randint(0, len(lst) - 1)
        newLst.append(lst[ind])
        del lst[ind]
    return newLst

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file,augmented_data_copies=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  data_augmentation_examples = [[example,0] for example in examples]

  for i in range(augmented_data_copies):
      data_augmentation_examples.extend([[example,1] for example in examples])
  data_augmentation_examples = shuffle(data_augmentation_examples)
  for (ex_index, [example,augment]) in enumerate(data_augmentation_examples):
    if ex_index % 10000 == 0:
      tf.logging.info(f"Writing example {ex_index} of {len(examples)}")

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer,create_altered_data=augment)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f


    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    if feature.ex_data:
        features["ex_data"] = create_float_feature(feature.ex_data)
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  examples = shuffle(examples,"examples")
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, shards_folder=None, pred_num=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  if pred_num:
      name_to_features = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.FixedLenFeature([], tf.int64),
          "ex_data": tf.FixedLenFeature([pred_num],tf.float32),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
      }
  else:
      name_to_features = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.FixedLenFeature([], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
      }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if shards_folder:
        import re
        file_name = input_file.split("/")[-1]
        shards = [shards_folder + "/" + file for file in tf.io.gfile.listdir(shards_folder) if
                  re.match(file_name + "_\d+", file)]
        shards = sorted(shards,key=lambda shard:int(shard.split("_")[-1]))
        print("USING SHARDS:")
        for shard in shards:
            print(shard)
        d = tf.data.Dataset.from_tensor_slices(shards)
    else:
        d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, model, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,weights,using_ex_data=False,ex_data=None):
  """Creates a classification model."""
  model = model(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  if using_ex_data:
      with tf.variable_scope("extra_data_layers"):
          pred_layer = tf.layers.dense(
                    ex_data,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=modeling.create_initializer(bert_config.initializer_range),
                    name="pred_dense")
          combined_layer = tf.concat([output_layer,pred_layer],axis=-1)
          output_layer = tf.layers.dense(
                    combined_layer,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=modeling.create_initializer(bert_config.initializer_range),
                    name="combine_dense")


  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    tot_loss = 0
    for l in range(0,num_labels):
        per_class_loss = -tf.reduce_sum(one_hot_labels[:,l] * log_probs[:,l], axis=-1)
        class_loss = tf.reduce_mean(per_class_loss)
        tot_loss+=class_loss*weights[l]

    return (tot_loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint,restore_checkpoint, init_learning_rate,
                     decay_per_step, num_warmup_steps, use_tpu, use_one_hot_embeddings, save_logs_every_n_steps=1, weights=None,freezing_x_layers=None,
                     yield_predictions=False,bert=modeling.BertModel, logging_dir=None,test_results_dir=None,weight_decay = 0.01,
                     epsilon=1e-4,optim="adam",clip_grads=True,using_ex_data = False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    if weights is None:
        class_weights = [1 for i in range(0,num_labels)]
    else:
        class_weights = weights

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    if using_ex_data:
        ex_data = features["ex_data"]
    else:
        ex_data = None


    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, bert, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings,class_weights,using_ex_data=using_ex_data,ex_data=ex_data)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if use_tpu:
        if init_checkpoint:
            if not restore_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                in_chkpt = [var for var in tvars if var.name in initialized_variable_names]
                def tpu_scaffold():
                    pretrain_saver = tf.train.Saver(var_list=in_chkpt)

                    def init_fn(scaffold, session):
                        pretrain_saver.restore(session, init_checkpoint)

                    return tf.train.Scaffold(init_fn=init_fn)
            else:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, restore_checkpoint)
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(restore_checkpoint, assignment_map)
                    return tf.train.Scaffold()
            scaffold_fn = tpu_scaffold

        elif restore_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, restore_checkpoint)
            def tpu_scaffold():
                tf.train.init_from_checkpoint(restore_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
    else:
        pass ##it will error saying scaffold_fn is not defined, because tpu is currently required

    if freezing_x_layers is not None:
        if freezing_x_layers=="all":
            freezing_layers = bert_config.num_hidden_layers
        else:
            freezing_layers = freezing_x_layers
        not_frozen = tvars[tvars.index([var for var in tvars if "encoder/layer_"+str(freezing_layers-1) in var.name][-1])+1:]
        grad_mask = [1 if var in not_frozen else 0 for var in tvars]

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars if not freezing_x_layers else not_frozen:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"

        else:
            init_string = ", *INIT_NEW*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

    global_step = tf.train.get_or_create_global_step()

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op, learning_rate = optimization.create_optimizer(
            total_loss, init_learning_rate, decay_per_step,
            num_warmup_steps, use_tpu, grad_mask = grad_mask if freezing_x_layers else None,
            weight_decay=weight_decay,epsilon=epsilon,optimizer_name=optim,clip=clip_grads)

        def train_metrics(ids, logits):
            """Computes the loss and accuracy of the model."""
            logits = tf.nn.softmax(tf.reshape(logits, [-1, logits.shape[-1]]),
                                       axis=-1)

            ids_1hot = tf.one_hot(tf.cast(ids, tf.int32), depth=num_labels, axis=-1)
            ids_1hot = tf.reshape(ids_1hot, [-1, ids_1hot.shape[-1]])
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predictions_1hot = tf.one_hot(tf.cast(predictions, tf.int32),
                                              depth=num_labels, axis=-1)
            ids_int = tf.reshape(ids, [-1])

            dice_f1_div = metric_functions.multiclass_f1_dice(logits, ids_1hot)
            recall_div = metric_functions.multiclass_recall(predictions_1hot, ids_1hot)
            precision_div = metric_functions.multiclass_precision(predictions_1hot, ids_1hot)
            acc_div = metric_functions.acc(predictions, ids_int)

            dice_f1 = dice_f1_div[0] / dice_f1_div[1]
            recall = recall_div[0] / recall_div[1]
            precision = precision_div[0] / precision_div[1]
            acc = acc_div[0] / acc_div[1]

            return {
                "dice_f1": dice_f1,
                "recall": recall,
                "precision": precision,
                "accuracy": acc
            }

        metrics = train_metrics(label_ids, logits)
        if logging_dir:
            def host_call_fn(gs, loss, lr, acc, prec, recall, f1):
                gs = gs[0]
                writer = tf.contrib.summary.create_file_writer(logging_dir)
                writer.set_as_default()

                def writing():
                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('loss', loss[0], step=gs)
                        tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
                        tf.contrib.summary.scalar('accuracy', acc[0], step=gs)
                        tf.contrib.summary.scalar('precision', prec[0], step=gs)
                        tf.contrib.summary.scalar('recall', recall[0], step=gs)
                        tf.contrib.summary.scalar('multiclass_averaged_dice/f1', f1[0], step=gs)
                    return tf.contrib.summary.all_summary_ops()

                def not_writing():
                    return [tf.constant(True) for _ in range(0, 6)]

                return tf.cond(
                    tf.equal(tf.mod(tf.cast(gs, tf.int32), tf.constant(save_logs_every_n_steps)), tf.constant(0)),
                    writing, not_writing)

            gs_t = tf.reshape(global_step, [1])
            loss_t = tf.reshape(total_loss, [1])
            lr_t = tf.reshape(learning_rate, [1])
            acc_t = tf.reshape(metrics["accuracy"], [1])
            precision_t = tf.reshape(metrics["precision"], [1])
            recall_t = tf.reshape(metrics["recall"], [1])
            f1_t = tf.reshape(metrics["dice_f1"], [1])

            host_call = (host_call_fn, [gs_t, loss_t, lr_t, acc_t, precision_t, recall_t, f1_t])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                host_call=host_call)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        
    elif mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(ids, logits):
            """Computes the loss and accuracy of the model."""
            logits = tf.nn.softmax(tf.reshape(logits, [-1, logits.shape[-1]]),
                                       axis=-1)

            ids_1hot = tf.one_hot(tf.cast(ids, tf.int32), depth=num_labels, axis=-1)
            ids_1hot = tf.reshape(ids_1hot, [-1, ids_1hot.shape[-1]])
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predictions_1hot = tf.one_hot(tf.cast(predictions, tf.int32),
                                              depth=num_labels, axis=-1)
            ids_int = tf.reshape(ids, [-1])

            accuracy = tf.metrics.accuracy(
                labels=ids_int,
                predictions=predictions, name="acc")

            AUC = tf.metrics.auc(
                labels=ids_int,
                predictions=logits[:,1], name="auc")

            dice = metric_functions.custom_metric(ids_1hot, logits,
                                                  custom_func=metric_functions.multiclass_f1_dice,
                                                  name="dice_f1")
            precision = metric_functions.custom_metric(ids_1hot, predictions_1hot,
                                                       custom_func=metric_functions.multiclass_precision,
                                                       name="multiclass_precision")
            recall = metric_functions.custom_metric(ids_1hot, predictions_1hot,
                                                    custom_func=metric_functions.multiclass_recall,
                                                    name="recall_multiclass")

            return {
                "accuracy": accuracy,
                "multiclass dice/f1": dice,
                "precision": precision,
                "recall": recall,
                "ROC AUC":AUC
            }

        eval_metrics = (metric_fn, [label_ids, logits])

        if yield_predictions:
            def host_call_fn(probs, label, inputids):
                with tf.contrib.summary.create_file_writer(test_results_dir).as_default():
                    with tf.contrib.summary.always_record_summaries():
                        for n in range(0, probs.shape.as_list()[0]):
                            tf.contrib.summary.scalar('probability', probs[n][1], step=n)
                            tf.contrib.summary.scalar('label', label[n], step=n)

                        return tf.contrib.summary.all_summary_ops()

            host_call = (host_call_fn, [probabilities, label_ids, input_ids])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                eval_metrics=eval_metrics,
                loss=total_loss,
                scaffold_fn=scaffold_fn,
                host_call=host_call)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                eval_metrics=eval_metrics,
                loss=total_loss,
                scaffold_fn=scaffold_fn)

    else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities,
                         "labels": label_ids},
            scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn