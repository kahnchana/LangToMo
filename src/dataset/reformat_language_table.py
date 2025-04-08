import glob
import multiprocessing as mp

import tensorflow as tf
import tensorflow_hub as hub
import tqdm


def generate_embedding(model, sentence_list):
    return model(sentence_list)


def _parse_function(proto):
    # Define the feature description dictionary
    # action_length = 14
    feature_description = {
        "steps/observation/rgb": tf.io.VarLenFeature(tf.string),
        "steps/action": tf.io.VarLenFeature(tf.float32),
        "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
        "steps/is_first": tf.io.VarLenFeature(tf.int64),
        "steps/is_last": tf.io.VarLenFeature(tf.int64),
        "steps/observation/instruction": tf.io.VarLenFeature(tf.int64),
        "steps/reward": tf.io.VarLenFeature(tf.float32),
    }
    return tf.io.parse_single_example(proto, feature_description)


def decode_instruction(record):
    """Current not used."""
    instruction_sparse = record["steps/observation/instruction"]
    instruction_dense = tf.sparse.to_dense(instruction_sparse)
    valid_codepoints = tf.boolean_mask(instruction_dense, instruction_dense != 0)
    valid_codepoints = tf.cast(valid_codepoints, tf.int32)
    instruction_text = tf.strings.unicode_encode(valid_codepoints, output_encoding="UTF-8")
    return instruction_text


def decode_all_step_instructions(record):
    instruction_sparse = record["steps/observation/instruction"]
    instruction_dense = tf.sparse.to_dense(instruction_sparse)
    dense_np = instruction_dense.numpy()

    tokens_per_instruction = 512
    num_steps = dense_np.shape[0] // tokens_per_instruction

    instruction_per_step = dense_np.reshape(num_steps, tokens_per_instruction)

    instruction_texts = []
    for tokens in instruction_per_step:
        tokens = tokens[tokens != 0]
        tokens = tf.cast(tokens, tf.int32)
        text = tf.strings.unicode_encode(tokens, output_encoding="UTF-8")
        instruction_texts.append(text)

    return instruction_texts


def add_embedding_to_record(record, model):
    instruction_texts = decode_all_step_instructions(record)
    embeddings = generate_embedding(model, instruction_texts)
    record["steps/observation/language_embedding"] = embeddings
    return record


def to_dense_numpy(tensor):
    if isinstance(tensor, tf.SparseTensor):
        tensor = tf.sparse.to_dense(tensor)
    return tensor.numpy()


def serialize_example(record):
    rgb_bytes = list(to_dense_numpy(record["steps/observation/rgb"]))
    instruction_ids = list(to_dense_numpy(record["steps/observation/instruction"]))
    reward_values = list(to_dense_numpy(record["steps/reward"]))
    action_values = list(to_dense_numpy(record["steps/action"]))
    is_terminal_val = list(to_dense_numpy(record["steps/is_terminal"]))
    is_first_val = list(to_dense_numpy(record["steps/is_first"]))
    is_last_val = list(to_dense_numpy(record["steps/is_last"]))
    lang_embedding = record["steps/observation/language_embedding"].numpy()
    lang_embedding = lang_embedding.flatten().tolist()

    feature = {
        "steps/observation/rgb": tf.train.Feature(bytes_list=tf.train.BytesList(value=rgb_bytes)),
        "steps/observation/instruction": tf.train.Feature(int64_list=tf.train.Int64List(value=instruction_ids)),
        "steps/observation/language_embedding": tf.train.Feature(float_list=tf.train.FloatList(value=lang_embedding)),
        "steps/action": tf.train.Feature(float_list=tf.train.FloatList(value=action_values)),
        "steps/is_terminal": tf.train.Feature(int64_list=tf.train.Int64List(value=is_terminal_val)),
        "steps/is_first": tf.train.Feature(int64_list=tf.train.Int64List(value=is_first_val)),
        "steps/is_last": tf.train.Feature(int64_list=tf.train.Int64List(value=is_last_val)),
        "steps/reward": tf.train.Feature(float_list=tf.train.FloatList(value=reward_values)),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def process_file_range(args):
    file_range, worker_idx = args
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    for cur_file in file_range:
        raw_dataset = tf.data.TFRecordDataset(cur_file)
        cur_name = cur_file.split("/")[-1]
        output_tfrecord = f"/nfs/ws2/kanchana/openx/language_table/0.0.1/{cur_name}"

        with tf.io.TFRecordWriter(output_tfrecord) as writer:
            for raw_record in tqdm.tqdm(raw_dataset, desc=cur_name, position=worker_idx, leave=False):
                parsed_record = _parse_function(raw_record)
                enriched_record = add_embedding_to_record(parsed_record, use_model)
                serialized_example = serialize_example(enriched_record)
                writer.write(serialized_example)

        print(f"Finished {cur_file}")


def chunkify(lst, n):
    """Split list `lst` into `n` roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    tfrecord_path = "/nfs/mercedes/hdd1/rt-x/language_table/0.0.1/language_table-train.tfrecord-*"
    tfrecord_files = sorted(glob.glob(tfrecord_path))

    num_workers = 16
    file_chunks = chunkify(tfrecord_files, num_workers)

    with mp.Pool(num_workers) as pool:
        pool.map(process_file_range, [(chunk, i) for i, chunk in enumerate(file_chunks)])

    DEBUG = False
    if DEBUG:
        # DEBUG CODE
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        total = len(tfrecord_files)

        i = 0
        for cur_file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(cur_file)
            output_tfrecord = f"/nfs/ws2/kanchana/openx/language_table/0.0.1/{cur_file.split('/')[-1]}"

            with tf.io.TFRecordWriter(output_tfrecord) as writer:
                for raw_record in tqdm.tqdm(raw_dataset):
                    parsed_record = _parse_function(raw_record)
                    enriched_record = add_embedding_to_record(parsed_record, use_model)
                    serialized_example = serialize_example(enriched_record)
                    writer.write(serialized_example)

            i += 1
            print(f"Processed {i} of {total} files.")
            if i > 10:
                break
