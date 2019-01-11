import tensorflow as tf


class Model:
    def __init__(self):
        pass

    def __call__(self, features, labels, mode, params):
        net = tf.feature_column.input_layer(features, params.feature_columns)
        for i, units in enumerate(params.hidden_units):
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu, name=f"dense_layer_{i}")

        logits = tf.layers.dense(net, params.n_classes, name='logits')
        predicted_classes = tf.argmax(logits, axis=-1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'prob': tf.nn.softmax(logits)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name="acc")
        metrics = {'accuracy': accuracy}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdamOptimizer(learning_rate=params.lr)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
