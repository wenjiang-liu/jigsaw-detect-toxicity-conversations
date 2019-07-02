import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
import tensorflow as tf
from tensorflow.keras import backend as K
from bert_fine_tune import *
from sklearn.model_selection import train_test_split
# from tensorflow.keras import regularizers
# from tensorflow.keras.utils import multi_gpu_model

# Params
TRAIN_INPUT = "data/concat_train.csv"
TEST_INPUT = "data/concat_test.csv"
IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
max_seq_length = 256
model_name = "BertModel_finetune_epoch1_processed_weights_aux.h5"


def build_model(max_seq_length, num_aux_targets):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    # uncomment following code to avoid overfitting
    """
    dense = tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l2(0.01))(bert_output)
    batchnormalization = tf.keras.layers.BatchNormalization()(dense)
    activation = tf.keras.layers.Activation('relu')(batchnormalization)
    dropout = tf.keras.layers.Dropout(0.5)(activation)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    """

    dense = tf.keras.layers.Dense(128, activation='relu')(bert_output)

    result = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    aux_result = tf.keras.layers.Dense(num_aux_targets, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=[result, aux_result])

    # parallel_model = multi_gpu_model(model, gpus=num_gpu)
    parallel_model = model
    parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return parallel_model, model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def get_weights(train_df):
    for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
        train_df[column] = np.where(train_df[column] >= 0.5, True, False)

    sample_weights = np.ones(len(train_df.index), dtype=np.float32)
    sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)  # increase weights of idaentity entries
    sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(
        axis=1)  # assign extra weights for toxicity comments without metioning a identity
    sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(
        axis=1) * 5  # assign extra weights for nontoxicity comments metioning a identity
    sample_weights /= sample_weights.mean()
    return sample_weights


def train(input_path_file):
    df = pd.read_csv(input_path_file)

    X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['target'], test_size=0.2,
                                                        random_state=42)
    X_train, X_test, y_aux_train, y_aux_test = train_test_split(df['comment_text'], df[IDENTITY_COLUMNS], test_size=0.2,
                                                                random_state=42)

    # Create datasets (Only take up to max_seq_length words for memory)
    train_df = pd.concat([pd.DataFrame({TEXT_COLUMN: X_train, TARGET_COLUMN: y_train}), y_aux_train], axis=1)

    sample_weights = get_weights(train_df)

    train_text = X_train.tolist()
    train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]  # take first 256 words，type is string
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]  # convert each element into [element]
    train_label = y_train.tolist()

    test_text = X_test.tolist()
    test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = y_test.tolist()

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer,
                                                                                                         train_examples,
                                                                                                         max_seq_length=max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer,
                                                                                                     test_examples,
                                                                                                     max_seq_length=max_seq_length)
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    for column in IDENTITY_COLUMNS:
        y_aux_train[column] = np.where(y_aux_train[column] >= 0.5, True, False)
        y_aux_test[column] = np.where(y_aux_test[column] >= 0.5, True, False)

    # parallel_model,model = build_model(max_seq_length,len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
    parallel_model, model = build_model(max_seq_length, y_aux_train.shape[-1])

    # Instantiate variables
    initialize_vars(sess)
    print("start fitting")

    parallel_model.fit([train_input_ids, train_input_masks, train_segment_ids], [train_labels, y_aux_train.values],
                       validation_data=(
                           [test_input_ids, test_input_masks, test_segment_ids], [test_labels, y_aux_test.values]),
                       sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
                       epochs=1, batch_size=64)

    model.save(model_name)


def test(input_path_file):
    df = pd.read_csv(input_path_file, encoding='utf8')
    tokenizer = create_tokenizer_from_hub_module()

    X_test = df['comment_text']

    test_text = X_test.tolist()
    test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = [None] * len(X_test)

    test_examples = convert_text_to_examples(test_text, test_label)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer,
                                                                                                     test_examples,
                                                                                                     max_seq_length=max_seq_length)
    model = build_model(max_seq_length, len(IDENTITY_COLUMNS))[1]
    initialize_vars(sess)
    model.load_weights(model_name)
    post_save_preds = model.predict([test_input_ids, test_input_masks, test_segment_ids])[0]
    df["target"] = post_save_preds
    df.to_csv("submission_utf8_aux.csv", index=False, encoding='utf8')
    df.to_excel("submission_utf8.xlsx", index=False, encoding='utf8')


if __name__ == "__main__":
    train(TRAIN_INPUT)
    test(TEST_INPUT)
