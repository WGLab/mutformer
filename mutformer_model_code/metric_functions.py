from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.
  If running in a `DistributionStrategy` context, the variable will be
  "sync on read". This means:
  *   The returned object will be a container with separate variables
      per replica of the model.
  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica.
  *   To get a metric's result value, we need to sum the variable values
      across the replicas before computing the final answer. Furthermore,
      the final answer should be computed once instead of in every
      replica. Both of these are accomplished by running the computation
      of the final result value inside
      `distribution_strategy_context.get_replica_context().merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a sync on read variable in a computation returns
      the sum across all replicas.
  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.
  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a sync on read variable container.
  """
  # Note that synchronization "ON_READ" implies trainable=False.
  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      trainable=False,
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      synchronization=variable_scope.VariableSynchronization.ON_READ,
      aggregation=variable_scope.VariableAggregation.SUM,
      name=name)

def _remove_squeezable_dimensions(predictions, labels, weights):
    """Squeeze or expand last dim if needed.
    Squeezes last dim of `predictions` or `labels` if their rank differs by 1
    (using confusion_matrix.remove_squeezable_dimensions).
    Squeezes or expands last dim of `weights` if its rank differs by 1 from the
    new rank of `predictions`.
    If `weights` is scalar, it is kept scalar.
    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.
    Args:
      predictions: Predicted values, a `Tensor` of arbitrary dimensions.
      labels: Optional label `Tensor` whose dimensions match `predictions`.
      weights: Optional weight scalar or `Tensor` whose dimensions match
        `predictions`.
    Returns:
      Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
      the last dimension squeezed, `weights` could be extended by one dimension.
    """
    predictions = ops.convert_to_tensor(predictions)
    if labels is not None:
        labels, predictions = confusion_matrix.remove_squeezable_dimensions(
            labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is None:
        return predictions, labels, None

    weights = ops.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return predictions, labels, weights

    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if (predictions_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - predictions_rank == 1:
            weights = array_ops.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = array_ops.expand_dims(weights, [-1])
    else:
        # Use dynamic rank.
        weights_rank_tensor = array_ops.rank(weights)
        rank_diff = weights_rank_tensor - array_ops.rank(predictions)

        def _maybe_expand_weights():
            return control_flow_ops.cond(
                math_ops.equal(rank_diff, -1),
                lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

        # Don't attempt squeeze if it will fail based on static check.
        if ((weights_rank is not None) and
                (not weights_shape.dims[-1].is_compatible_with(1))):
            maybe_squeeze_weights = lambda: weights
        else:
            maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            return control_flow_ops.cond(
                math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
                _maybe_expand_weights)

        # If weights are scalar, do nothing. Otherwise, try to add or remove a
        # dimension to match predictions.
        weights = control_flow_ops.cond(
            math_ops.equal(weights_rank_tensor, 0), lambda: weights,
            _maybe_adjust_weights)
    return predictions, labels, weights

def multiclass_f1_dice(predictions, labels, weights=None):
    smooth = 0.000001
    if weights is None:
        weights = tf.ones(predictions.shape[:-1])
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=predictions,
        labels=labels,
        weights=weights)
    print(predictions.shape,labels.shape,weights.shape)
    tot_dice = 0
    for l in range(0, predictions.shape.as_list()[-1]):
        preds = tf.reshape(tf.slice(predictions,[0,l],[predictions.shape.as_list()[0],1]),[-1])
        lbls = tf.reshape(tf.slice(labels,[0,l],[labels.shape.as_list()[0],1]),[-1])
        intersect = math_ops.reduce_sum(preds * lbls * weights)
        both = math_ops.reduce_sum((preds + lbls) * weights)
        dice = ((2 * intersect) + smooth) / (both + smooth)
        tot_dice += dice
    return tot_dice, tf.constant(predictions.shape.as_list()[-1],dtype=tf.float32)

def acc(predictions, labels, weights=None): ## just normal accuracy (TP + TN)/(TP + TN + FP + FN); total_correct/total
    smooth = 0.000001
    if weights is None:
        weights = tf.ones(predictions.shape)
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=predictions,
        labels=labels,
        weights=weights)
    correct = tf.cast(math_ops.equal(predictions,labels),tf.float32)
    correct=tf.reduce_sum(correct*weights)
    tot = tf.reduce_sum(weights)
    print("acctot:",tot)
    return correct+smooth,tot+smooth

def multiclass_recall(predictions, labels, weights=None):
    smooth = 0.000001
    if weights is None:
        weights = tf.ones(predictions.shape[:-1])
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    tot_recall=0
    for l in range(0, predictions.shape.as_list()[-1]):
        preds = tf.reshape(tf.slice(predictions,[0,l],[predictions.shape.as_list()[0],1]),[-1])
        lbls = tf.reshape(tf.slice(labels,[0,l],[labels.shape.as_list()[0],1]),[-1])
        is_true_positive = math_ops.logical_and(math_ops.equal(lbls, True),math_ops.equal(preds, True))
        tp = math_ops.reduce_sum(tf.cast(is_true_positive, tf.float32)*weights)

        is_false_negative = math_ops.logical_and(math_ops.equal(lbls, True),math_ops.equal(preds, False))
        fn = math_ops.reduce_sum(tf.cast(is_false_negative, tf.float32)*weights)

        recall = (tp+smooth)/(tp+fn+smooth)
        tot_recall+=recall
    return tot_recall,tf.constant(predictions.shape.as_list()[-1],dtype=tf.float32)

def multiclass_precision(predictions, labels, weights=None):
    smooth = 0.000001
    if weights is None:
        weights = tf.ones(predictions.shape[:-1])
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions=math_ops.cast(predictions, dtype=dtypes.bool),
        labels=math_ops.cast(labels, dtype=dtypes.bool),
        weights=weights)
    tot_precision=0
    for l in range(0, predictions.shape.as_list()[-1]):
        preds = tf.reshape(tf.slice(predictions, [0, l], [predictions.shape.as_list()[0], 1]), [-1])
        lbls = tf.reshape(tf.slice(labels, [0, l], [labels.shape.as_list()[0], 1]), [-1])

        is_true_positive = math_ops.logical_and(math_ops.equal(lbls, True),math_ops.equal(preds, True))
        tp = math_ops.reduce_sum(tf.cast(is_true_positive, tf.float32)*weights)

        is_false_positive = math_ops.logical_and(math_ops.equal(lbls, False),math_ops.equal(preds, True))
        fp = math_ops.reduce_sum(tf.cast(is_false_positive, tf.float32)*weights)

        precision = (tp+smooth)/(tp+fp+smooth)
        tot_precision+=precision
    return tot_precision,tf.constant(predictions.shape.as_list()[-1],dtype=tf.float32)


def custom_metric(labels,
                  predictions,
                  custom_func,
                  weights=None,
                  metrics_collections=None,
                  updates_collections=None,
                  name=None):
    with variable_scope.variable_scope(name, 'custom_metric', (labels, predictions, weights)):
        labels = math_ops.cast(labels, dtypes.float32)
        predictions = math_ops.cast(predictions, dtypes.float32)

        total = metric_variable([], dtypes.float32, name='total')
        count = metric_variable([], dtypes.float32, name='count')

        num, denom = custom_func(labels, predictions, weights)

        update_total_op = state_ops.assign_add(total, num)
        with ops.control_dependencies([denom]):
            update_count_op = state_ops.assign_add(count, denom)

        def compute_metric(_, t, c):
            return math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')

        metric_t = _aggregate_across_replicas(
            metrics_collections, compute_metric, total, count)
        update_op = math_ops.div_no_nan(
            update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return metric_t, update_op


def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across replicas."""

    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution.extended, '_outer_control_flow_context'):
            # If there was an outer context captured before this method was called,
            # then we enter that context to create the metric value op. If the
            # captured context is `None`, ops.control_dependencies(None) gives the
            # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
            # captured context.
            # This special handling is needed because sometimes the metric is created
            # inside a while_loop (and perhaps a TPU rewrite context). But we don't
            # want the value op to be evaluated every step or on the TPU. So we
            # create it outside so that it can be evaluated at the end on the host,
            # once the update ops have been evaluated.

            # pylint: disable=protected-access
            if distribution.extended._outer_control_flow_context is None:
                with ops.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution.extended._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution.extended._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            ops.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return distribution_strategy_context.get_replica_context().merge_call(
        fn, args=args)