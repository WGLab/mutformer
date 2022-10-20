from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf
import internals.modeling as modeling


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file for running inference (or comma-separated list of files) (should be formatted corresponding to the test mode chosen from the Finetuning section in the README).")

flags.DEFINE_string(
    "output_file", None,
    "File to output inference results to (or comma-separated list of files).")

flags.DEFINE_string("model_folder", None,
                    "Folder where the model checkpoint and config file is stored.")

flags.DEFINE_string("model_architecture", None,
                    "Model architecture of the model checkpoint specified (BertModel indicates the original BERT, BertModelModified indicates MutFormer's architecture without integrated convs, MutFormer_embedded_convs indicates MutFormer with integrated convolutions).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("batch_size", 64, "Number of lines to batch together at a time when performing inference.")


def compile_single_example(line):
    tokens = ["[CLS]"]
    segment_ids = [0]

    line_components = line.split("\t")

    for char in line_components[3].split(" "):
        tokens.append(char)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for char in line_components[4].split(" "):
        tokens.append(char)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_mask = [1 for x in tokens]
    while len(tokens) < 512:
        tokens.append("[PAD]")
        segment_ids.append(0)
        input_mask.append(0)
    tokens = [vocab.index(x) for x in tokens]
    return tokens, segment_ids, input_mask


def run_model_for_probabilities(config,
                                tokens,
                                masks,
                                segment_ids,
                                model2use):
    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    init = tf.global_variables_initializer()

    tokens = tf.constant(tokens)
    masks = tf.constant(masks)
    segment_ids = tf.constant(segment_ids)

    model = model2use(
        config=config,
        is_training=False,
        input_ids=tokens,
        input_mask=masks,
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

    with tf.Session() as sess:
        sess.run(init)
        run_probabilities = sess.run(probabilities)
    tf.reset_default_graph()
    return run_probabilities


init_checkpoint = tf.train.latest_checkpoint(FLAGS.model_folder)
vocab = open(FLAGS.vocab_file).read().split("\n")
config = json.load(open(FLAGS.model_folder + "/config.json"))
config = modeling.BertConfig.from_dict(config)

num_labels = 2

with open(FLAGS.output_file, "w+") as outf:
    with open(FLAGS.input_file) as inf:
        EOF = False
        batch_num = 0
        while not EOF:
            print(f"Preparing to process batch number {batch_num} of size {FLAGS.batch_size}...")

            tokens_list = []
            segment_ids_list = []
            input_mask_list = []
            for i in range(0, FLAGS.batch_size):
                line = inf.readline()
                if not line:
                    print("\nReached end of input file. Processing last batch")
                    EOF = True
                    break
                tokens, segment_ids, input_mask = compile_single_example(line)
                tokens_list.append(tokens)
                segment_ids_list.append(segment_ids)
                input_mask_list.append(input_mask)
            probabilities = run_model_for_probabilities(config=config,
                                                        tokens=tokens_list,
                                                        segment_ids=segment_ids_list,
                                                        masks=input_mask_list,
                                                        model2use=getattr(modeling,FLAGS.model_architecture))
            outf.write("\n".join([str(x) for x in probabilities]) + "\n")
            batch_num += 1
