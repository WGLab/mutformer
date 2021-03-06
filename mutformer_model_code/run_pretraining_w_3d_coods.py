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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import metric_functions

def model_fn_builder(bert_config, init_checkpoint, init_learning_rate,
                     decay_per_step, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, save_logs_every_n_steps=1, logging_dir=None, bert=modeling.MutFormer_distance_scaled_context):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    coodss_x = features["3dcood_x"]
    coodss_y = features["3dcood_y"]
    coodss_z = features["3dcood_z"]

    distance_map,coodss_all = process_coods(coodss_x,coodss_y,coodss_z,bert_config)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = bert(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        coods=coodss_all,
        distance_map=distance_map,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss, masked_lm_log_probs, masked_lm_logits) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    total_loss = masked_lm_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    global_step = tf.train.get_or_create_global_step()

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op,learning_rate = optimization.create_optimizer(
            total_loss, init_learning_rate, decay_per_step, num_warmup_steps, use_tpu)

        def train_metrics(masked_lm_log_probs, masked_lm_ids, masked_lm_weights, masked_lm_logits):
            """Computes the loss and accuracy of the model."""
            masked_lm_logits = tf.nn.softmax(tf.reshape(masked_lm_logits, [-1, masked_lm_logits.shape[-1]]), axis=-1)

            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            masked_lm_ids_1hot = tf.one_hot(tf.cast(masked_lm_ids, tf.int32), depth=bert_config.vocab_size, axis=-1)
            masked_lm_ids_1hot = tf.reshape(masked_lm_ids_1hot, [-1, masked_lm_ids_1hot.shape[-1]])
            masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
            masked_lm_predictions_1hot = tf.one_hot(tf.cast(masked_lm_predictions, tf.int32), depth=bert_config.vocab_size, axis=-1)
            masked_lm_ids_int = tf.reshape(masked_lm_ids, [-1])

            dice_f1_div = metric_functions.multiclass_f1_dice(masked_lm_logits,masked_lm_ids_1hot,masked_lm_weights)
            recall_div = metric_functions.multiclass_recall(masked_lm_predictions_1hot, masked_lm_ids_1hot, masked_lm_weights)
            precision_div = metric_functions.multiclass_precision(masked_lm_predictions_1hot, masked_lm_ids_1hot, masked_lm_weights)
            acc_div = metric_functions.acc(masked_lm_predictions, masked_lm_ids_int, masked_lm_weights)


            dice_f1 = dice_f1_div[0]/dice_f1_div[1]
            recall = recall_div[0]/recall_div[1]
            precision = precision_div[0]/precision_div[1]
            acc = acc_div[0]/acc_div[1]


            return {
                "dice_f1":dice_f1,
                "recall":recall,
                "precision":precision,
                "accuracy":acc
            }

        metrics = train_metrics(masked_lm_log_probs, masked_lm_ids, masked_lm_weights, masked_lm_logits)

        gs_t = tf.reshape(global_step, [1])

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
      def metric_fn(masked_lm_log_probs, masked_lm_ids, masked_lm_weights,masked_lm_logits):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,[-1, masked_lm_log_probs.shape[-1]])
        masked_lm_logits = tf.nn.softmax(tf.reshape(masked_lm_logits,[-1, masked_lm_logits.shape[-1]]),axis=-1)

        masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_predictions_1hot = tf.one_hot(tf.cast(masked_lm_predictions, tf.int32), depth=bert_config.vocab_size,axis=-1)
        masked_lm_ids_int = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_ids_1hot = tf.one_hot(tf.cast(masked_lm_ids,tf.int32), depth=bert_config.vocab_size,axis=-1)
        masked_lm_ids_1hot = tf.reshape(masked_lm_ids_1hot, [-1, masked_lm_ids_1hot.shape[-1]])

        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids_int,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights,name = "acc")

        dice=metric_functions.custom_metric(masked_lm_ids_1hot,masked_lm_logits,
                                            custom_func=metric_functions.multiclass_f1_dice,
                                            weights=masked_lm_weights,name="dice_f1")
        precision = metric_functions.custom_metric(masked_lm_ids_1hot, masked_lm_predictions_1hot,
                                                   custom_func=metric_functions.multiclass_precision,
                                                   weights=masked_lm_weights,name="multiclass_precision")
        recall = metric_functions.custom_metric(masked_lm_ids_1hot, masked_lm_predictions_1hot,
                                                custom_func=metric_functions.multiclass_recall,
                                                weights=masked_lm_weights,name="recall_multiclass")


        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "multiclass dice/f1": dice,
            "precision": precision,
            "recall": recall,
        }

      eval_metrics = (metric_fn, [masked_lm_log_probs,
                                  masked_lm_ids,
                                  masked_lm_weights,
                                  masked_lm_logits])


      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          eval_metrics=eval_metrics,
          loss=total_loss,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

def process_coods(coodss_x,coodss_y,coodss_z,bert_config):
    input_length = coodss_x.shape[1]

    def infer_from_coodset(input_tensor):
        invalid_mask = tf.cast(tf.equal(input_tensor, tf.constant(1e8)), tf.float32)
        invalid_mask_horiz = tf.broadcast_to(tf.expand_dims(invalid_mask, 1),
                                             [invalid_mask.shape[0], input_length, input_length])
        invalid_mask_vert = tf.broadcast_to(tf.expand_dims(invalid_mask, 2),
                                            [invalid_mask.shape[0], input_length, input_length])

        distance_map_invalid_mask = tf.cast(tf.greater_equal(invalid_mask_vert + invalid_mask_horiz, 1), tf.float32)

        centers = (tf.reduce_max(input_tensor * invalid_mask, axis=1) +
                   tf.reduce_min(input_tensor * invalid_mask, axis=1)) / 2

        coodss_2d_horiz = tf.broadcast_to(tf.expand_dims(input_tensor, 1),
                                          [input_tensor.shape[0], input_length, input_length])
        coodss_2d_vert = tf.broadcast_to(tf.expand_dims(input_tensor, 2),
                                         [input_tensor.shape[0], input_length, input_length])

        coodss_2d_vert = coodss_2d_vert - (invalid_mask_vert * 1e8)
        coodss_2d_horiz = coodss_2d_horiz - (invalid_mask_horiz * 1e8)

        coodss_distance_map = tf.abs(coodss_2d_horiz - coodss_2d_vert)
        return invalid_mask, distance_map_invalid_mask, centers, coodss_distance_map

    with  tf.variable_scope("3d_coods_processing"):
        ##distance map creation

        coods_mask_x, distance_mask_x, centers_x, coods_distances_x = infer_from_coodset(coodss_x)
        coods_mask_y, distance_mask_y, centers_y, coods_distances_y = infer_from_coodset(coodss_y)
        coods_mask_z, distance_mask_z, centers_z, coods_distances_z = infer_from_coodset(coodss_z)

        coods_distances_all = tf.stack([coods_distances_x,coods_distances_y,coods_distances_z],axis=3)

        distances_mask_x_wfinalshape = tf.broadcast_to(tf.expand_dims(distance_mask_x,3), [distance_mask_x.shape[0], input_length, input_length,3])
        distances_mask_y_wfinalshape = tf.broadcast_to(tf.expand_dims(distance_mask_y,3), [distance_mask_y.shape[0], input_length, input_length,3])
        distances_mask_z_wfinalshape = tf.broadcast_to(tf.expand_dims(distance_mask_z,3), [distance_mask_z.shape[0], input_length, input_length,3])

        distances_mask_all = tf.cast(tf.greater(distances_mask_x_wfinalshape+
                                                distances_mask_y_wfinalshape+
                                                distances_mask_z_wfinalshape,0),tf.float32)

        coods_distances_all = coods_distances_all*((distances_mask_all-1)*-1)

        distances_all = tf.sqrt(tf.reduce_sum(tf.square(coods_distances_all),axis=3))
        if not hasattr(bert_config, "multiplier_num"):
            power_var = tf.get_variable(
                "distance_power",
                shape=[1],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            power = tf.abs(power_var*100)
            distances_squared = tf.pow(distances_all,power)
            multiplier_num_var = tf.get_variable(
                "multiplier_num",
                shape=[1],
                initializer=modeling.create_initializer(bert_config.initializer_range))
            multiplier_num = tf.abs(multiplier_num_var*1000)
            distances_ready_for_division = distances_squared + (tf.cast(tf.equal(distances_squared, 0), tf.float32) *
                                                                multiplier_num)
            distance_map = multiplier_num/distances_ready_for_division
        else:
            distances_squared = tf.square(distances_all)
            distances_ready_for_division = distances_squared + (tf.cast(tf.equal(distances_squared, 0), tf.float32) *
                                                                bert_config.multiplier_num)
            distance_map = bert_config.multiplier_num / distances_ready_for_division

        ##coods creation
        coodss_x=(coodss_x-tf.broadcast_to(tf.expand_dims(centers_x,1),
                                           [centers_x.shape[0],input_length]))*coods_mask_x
        coodss_y=(coodss_x-tf.broadcast_to(tf.expand_dims(centers_y,1),
                                           [centers_y.shape[0],input_length]))*coods_mask_y
        coodss_z=(coodss_x-tf.broadcast_to(tf.expand_dims(centers_z,1),
                                           [centers_z.shape[0],input_length]))*coods_mask_z

        coodss_all = tf.stack([coodss_x,coodss_y,coodss_z],axis=2)
    return distance_map,coodss_all

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
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

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, log_probs, logits)



def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "3dcood_x":
            tf.FixedLenFeature([max_seq_length], tf.float32),
        "3dcood_y":
            tf.FixedLenFeature([max_seq_length], tf.float32),
        "3dcood_z":
            tf.FixedLenFeature([max_seq_length], tf.float32),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


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
