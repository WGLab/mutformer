from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow.compat.v1 as tf
import internals.modeling as modeling
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

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

flags.DEFINE_integer("max_seq_length", 1024, "Maximum sequence length to pad to; should be greater than or equal to the three more than twice the length of the longest input amino acid sequence.")


flags.DEFINE_boolean("use_ex_data", False,
                    "Whether or not input data was generated with external data included")


def compile_single_example(line):
    tokens = ["[CLS]"]
    segment_ids = [0]

    line_components = line.split("\t")

    for char in line_components[0].split(" "):
        tokens.append(char)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for char in line_components[1].split(" "):
        tokens.append(char)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_mask = [1 for x in tokens]
    while len(tokens) < FLAGS.max_seq_length:
        tokens.append("[PAD]")
        segment_ids.append(0)
        input_mask.append(0)
    tokens = [vocab.index(x) for x in tokens]
    return tokens, segment_ids, input_mask, line_components[2] if FLAGS.use_ex_data else None


def run_model_for_probabilities(config,
                                tokens,
                                masks,
                                ex_datas,
                                segment_ids,
                                model2use):

    tokens = tf.constant(tokens)
    masks = tf.constant(masks)
    segment_ids = tf.constant(segment_ids)

    def run_model():
        model = model2use(
            config=config,
            is_training=True,
            input_ids=tokens,
            input_mask=masks,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=True)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1]

        if FLAGS.use_ex_data:
            with tf.variable_scope("extra_data_layers"):
                pred_layer = tf.layers.dense(
                    ex_datas,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=modeling.create_initializer(config.initializer_range),
                    name="pred_dense")
                combined_layer = tf.concat([output_layer, pred_layer], axis=-1)
                output_layer = tf.layers.dense(
                    combined_layer,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=modeling.create_initializer(config.initializer_range),
                    name="combine_dense")

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        return tf.nn.softmax(logits, axis=-1)

    probabilities = run_model()

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    init = tf.global_variables_initializer()

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
            ex_data_list = []
            for i in range(0, FLAGS.batch_size):
                line = inf.readline().strip()
                if not line:
                    print("\nReached end of input file. Processing last batch")
                    EOF = True
                    break
                tokens, segment_ids, input_mask, ex_data = compile_single_example(line)
                tokens_list.append(tokens)
                segment_ids_list.append(segment_ids)
                input_mask_list.append(input_mask)
                ex_data_list.append(ex_data)
            probabilities = run_model_for_probabilities(config=config,
                                                        tokens=tokens_list,
                                                        segment_ids=segment_ids_list,
                                                        masks=input_mask_list,
                                                        ex_datas=ex_data_list,
                                                        model2use=getattr(modeling,FLAGS.model_architecture))
            print(f"Finished processing batch number {batch_num}. Writing to output file...")
            outf.write("\n".join([str(x) for x in probabilities]) + "\n")
            batch_num += 1
