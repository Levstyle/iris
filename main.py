import tensorflow as tf
from tools.prep import load_data, train_input_fn, eval_input_fn, SPECIES
from models.model import Model

tf.logging.set_verbosity(tf.logging.INFO)


def serving_input_fn():
    receiver_tensor = {
        'SepalLength': tf.placeholder(tf.float32, shape=(None,)),
        'SepalWidth': tf.placeholder(tf.float32, shape=(None,)),
        'PetalLength': tf.placeholder(tf.float32, shape=(None,)),
        'PetalWidth': tf.placeholder(tf.float32, shape=(None,))
    }

    features = {
        key: tensor for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


def main(*args, **kwargs):
    (train_x, train_y), (test_x, test_y) = load_data()
    feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

    hparams = tf.contrib.training.HParams(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=3, lr=1e-4,
                                          batch_size=64, train_steps=12000)

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1000,
        tf_random_seed=19830610,
        model_dir='./save'
    )

    classifier = tf.estimator.Estimator(model_fn=Model(), params=hparams, config=run_config)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(train_x, train_y, hparams.batch_size),
        max_steps=hparams.train_steps,
        hooks=None
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(test_x, test_y, hparams.batch_size),
        exporters=[tf.estimator.LatestExporter(
            name="predict",
            serving_input_receiver_fn=serving_input_fn,
            exports_to_keep=1,
            as_text=False)],
        steps=None,
        throttle_secs=20
    )

    tf.estimator.train_and_evaluate(
        estimator=classifier,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                       labels=None,
                                       batch_size=hparams.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['prob'][class_id]

        print(template.format(SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.app.run()
