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
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from tqdm import tqdm

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, pos, labels=None):
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
        self.labels = labels
        self.pos = pos


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 mutation_mask,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.mutation_mask = mutation_mask
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
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

class NERProcessor(DataProcessor):
    """Processor for the ner data set (GLUE version)."""

    def get_train_examples(self, data_dir, read_range=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), read_range=read_range), "train")

    def get_dev_examples(self, data_dir, read_range=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv"), read_range=read_range), "dev")

    def get_test_examples(self, data_dir, read_range=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), read_range=read_range), "test")

    def get_labels(self):
        """See base class."""
        return ["B", "P"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(tqdm(lines,"creating examples")):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            labels = tokenization.convert_to_unicode(line[1])
            pos = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a,pos=pos, labels=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            mutation_mask=[0] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    labels = example.labels.split(" ")
    labels = [label_map[lbl] for lbl in labels]
    position=int(example.pos)+1

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    tokens = []
    mut_mask = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    mut_mask.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
        mut_mask.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    mut_mask.append(0)

    mut_mask[position] = 1

    label_ids = []
    label_ids.append(-1)
    for label in labels:
        label_ids.append(label)
    label_ids.append(-1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-1)
        mut_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(mut_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("mutation mask: %s" % " ".join([str(x) for x in mut_mask]))
        tf.logging.info("position: %s" % str(position))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        mutation_mask=mut_mask,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["mutation_mask"] = create_int_feature(feature.mutation_mask)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder,shards_folder = None,pred_num=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mutation_mask": tf.FixedLenFeature([seq_length], tf.int64),
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
            shards = sorted(shards, key=lambda shard: int(shard.split("_")[-1]))
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


def create_model(bert_config, model, is_training, input_ids, input_mask, mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, weights):
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
    input_tensor = model.get_sequence_output()

    with tf.variable_scope("loss"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)
        if is_training:
            input_tensor = tf.nn.dropout(input_tensor, keep_prob=0.9)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, bert_config.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        tot_loss = 0
        for n in range(0,one_hot_labels.shape.as_list()[0]):
            for l in range(0, num_labels):
                class_loss = -tf.reduce_sum(one_hot_labels[n, :, l] * log_probs[n, :, l] * tf.cast(mask[n],tf.float32))
                tot_loss += class_loss * weights[l]

        return (tot_loss, logits, probabilities)



def model_fn_builder(bert_config, num_labels, init_checkpoint, restore_checkpoint, init_learning_rate,
                     decay_per_step, num_warmup_steps, use_tpu, use_one_hot_embeddings, weights=None, freezing=None,
                     yield_predictions=False, bert=modeling.BertModel, test_results_dir=None, weight_decay=0.01,
                     epsilon=1e-4, optim="adam", clip_grads=True,using_ex_data=False,logging_dir=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        if weights is None:
            class_weights = [1 for i in range(0, num_labels)]
        else:
            class_weights = weights

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        mutation_masks = features["mutation_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, probabilities) = create_model(
            bert_config, bert, is_training, input_ids, input_mask, mutation_masks,
            segment_ids, label_ids, num_labels, use_one_hot_embeddings, class_weights)

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
            pass  ##it will error saying scaffold_fn is not defined, because tpu is currently required

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"

            else:
                init_string = ", *INIT_NEW*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        global_step = tf.train.get_or_create_global_step()

        if freezing is not None:
            not_frozen = [v for v in tf.global_variables() if not "embedding" in v.name]
            if freezing == "all":
                freezing_layers = bert_config.num_hidden_layers
            else:
                freezing_layers = freezing
            for v in tf.global_variables():
                for i in range(0, freezing_layers):
                    if not "encoder/layer_" + str(i) in v.name and "conv" not in v.name:
                        not_frozen.append(v)

            tf.logging.info("TRAINING VARIABLES")
            for v in not_frozen:
                tf.logging.info(v.name)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, learning_rate = optimization.create_optimizer(
                total_loss, init_learning_rate, decay_per_step,
                num_warmup_steps, use_tpu, tvars=not_frozen if freezing else None,
                weight_decay=weight_decay, epsilon=epsilon, optimizer_name=optim, clip=clip_grads)

            def train_metrics(ner_ids, ner_logits, ner_mask):
                """Computes the loss and accuracy of the model."""
                ner_mask = tf.cast(tf.reshape(ner_mask, [-1]),tf.float32)
                ner_logits = tf.nn.softmax(tf.reshape(ner_logits, [-1, ner_logits.shape[-1]]),axis=-1)

                ner_ids_1hot = tf.one_hot(tf.cast(ner_ids, tf.int32), depth=num_labels, axis=-1)
                ner_ids_1hot = tf.reshape(ner_ids_1hot, [-1, ner_ids_1hot.shape[-1]])
                ner_predictions = tf.argmax(ner_logits, axis=-1, output_type=tf.int32)
                ner_predictions_1hot = tf.one_hot(tf.cast(ner_predictions, tf.int32),depth=num_labels, axis=-1)

                ner_ids_int = tf.reshape(ner_ids, [-1])

                dice_f1_div = metric_functions.multiclass_f1_dice(ner_logits, ner_ids_1hot,ner_mask)
                recall_div = metric_functions.multiclass_recall(ner_predictions_1hot, ner_ids_1hot,ner_mask)
                precision_div = metric_functions.multiclass_precision(ner_predictions_1hot, ner_ids_1hot,ner_mask)
                acc_div = metric_functions.acc(ner_predictions, ner_ids_int,ner_mask)

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

            metrics = train_metrics(label_ids, logits, mutation_masks)
            if logging_dir:
                print("USING logging_dir")
                def host_call_fn(gs, loss, lr, acc, prec, recall, f1):
                    with tf.contrib.summary.create_file_writer(logging_dir).as_default():
                        gs = gs[0]
                        with tf.contrib.summary.always_record_summaries():
                            tf.contrib.summary.scalar('train_loss', loss[0], step=gs)
                            tf.contrib.summary.scalar('learning_rate', lr[0], step=gs)
                            tf.contrib.summary.scalar('accuracy', acc[0], step=gs)
                            tf.contrib.summary.scalar('precision', prec[0], step=gs)
                            tf.contrib.summary.scalar('recall', recall[0], step=gs)
                            tf.contrib.summary.scalar('multiclass_averaged_dice/f1', f1[0], step=gs)

                            return tf.contrib.summary.all_summary_ops()

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

            def metric_fn(ner_ids, ner_logits, ner_mask):
                """Computes the loss and accuracy of the model."""
                ner_mask = tf.cast(tf.reshape(ner_mask, [-1]),tf.float32)
                ner_logits = tf.nn.softmax(tf.reshape(ner_logits, [-1, ner_logits.shape[-1]]),
                                            axis=-1)

                ner_ids_1hot = tf.one_hot(tf.cast(ner_ids, tf.int32), depth=num_labels, axis=-1)
                ner_ids_1hot = tf.reshape(ner_ids_1hot, [-1, ner_ids_1hot.shape[-1]])
                ner_predictions = tf.argmax(ner_logits, axis=-1, output_type=tf.int32)
                ner_predictions_1hot = tf.one_hot(tf.cast(ner_predictions, tf.int32),depth=num_labels, axis=-1)
                ner_ids_int = tf.reshape(ner_ids, [-1])


                accuracy = tf.metrics.accuracy(
                    labels=ner_ids_int,
                    predictions=ner_predictions,
                    weights=ner_mask,
                    name="acc")

                AUC = tf.metrics.auc(
                    labels=ner_ids_int,
                    predictions=ner_logits[:,1],
                    weights=ner_mask,
                    name="auc")

                dice = metric_functions.custom_metric(ner_logits, ner_ids_1hot,
                                                      custom_func=metric_functions.multiclass_f1_dice,
                                                      name="dice_f1",
                                                      weights=ner_mask)
                precision = metric_functions.custom_metric(ner_predictions_1hot, ner_ids_1hot,
                                                           custom_func=metric_functions.multiclass_precision,
                                                           name="multiclass_precision",
                                                           weights=ner_mask)
                recall = metric_functions.custom_metric(ner_predictions_1hot, ner_ids_1hot,
                                                        custom_func=metric_functions.multiclass_recall,
                                                        name="recall_multiclass",
                                                        weights=ner_mask)

                return {
                    "accuracy": accuracy,
                    "multiclass dice/f1": dice,
                    "precision": precision,
                    "recall": recall,
                    "ROC AUC": AUC
                }

            eval_metrics = (metric_fn, [label_ids, logits,mutation_masks])

            if yield_predictions:
                print("USING EVALUATE_WHILE_PREDICT")
                def host_call_fn(probs, labels, masks):
                    with tf.contrib.summary.create_file_writer(test_results_dir).as_default():
                        with tf.contrib.summary.always_record_summaries():
                            for n in range(0, probs.shape.as_list()[0]):
                                positive_class_probs = probs[n,:,1] * tf.cast(masks[n],tf.float32)
                                tf.contrib.summary.scalar('probabilities', tf.reduce_sum(positive_class_probs), step=n)
                                tf.contrib.summary.scalar('labels', tf.reduce_sum(tf.cast(labels[n],tf.float32) * tf.cast(masks[n],tf.float32)), step=n)

                            return tf.contrib.summary.all_summary_ops()

                host_call = (host_call_fn, [probabilities, label_ids, mutation_masks])

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
                predictions={"probabilities": tf.reduce_sum(probabilities[:,:,1] * tf.cast(mutation_masks,tf.float32),axis=1),
                             "labels": tf.reduce_sum(tf.cast(label_ids,tf.float32) * tf.cast(mutation_masks,tf.float32),axis=1)},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
