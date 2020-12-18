import helper
import os
import sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import itertools
from transformers import BertTokenizer
from enum import Enum, auto
from tensorflow.keras.models import Model
from collections import Counter
from typing import Dict, Any, Union, List


class EmbeddingMode(Enum):
    GLOVE = auto()
    BERT = auto()
    OTHER = auto()

    @classmethod
    def from_str(cls, string: str):
        lowered_string = string.lower()

        if lowered_string == 'glove':
            return EmbeddingMode.GLOVE
        elif lowered_string == 'bert':
            return EmbeddingMode.BERT
        else:
            return EmbeddingMode.OTHER

    def __str__(self):
        if self == EmbeddingMode.GLOVE:
            return 'glove'
        elif self == EmbeddingMode.BERT:
            return 'bert'
        elif self == EmbeddingMode.OTHER:
            return 'other'


sys.path.insert(1, os.path.join(sys.path[0], '..'))


##########################################################################
############################## Word Vocab Generation #####################
##########################################################################


def tokens2id(tokens, word2id, max_len):
    idx = []
    if tokens == ['']:
        for _ in range(max_len):
            idx.append(word2id["PAD"])
    else:
        for t in tokens[:max_len]:
            if t not in word2id.keys():
                idx.append(word2id["UNK"])
            else:
                idx.append(word2id[t])
        padding_needed = max_len - len(idx) if max_len > len(idx) else 0
        for _ in range(padding_needed):
            idx.append(word2id["PAD"])
    assert len(idx) == max_len
    return idx


def idx_representations_of_text(text_list, context_list, word2id, max_len, k):
    idx_data_list, splits = [], ["train", "valid", "test"]
    for i in range(k):
        idx_data = {}
        for split in splits:
            idx_data[split] = []
            for index, tokens in enumerate(text_list[i][split]):
                ctxt_idx = tokens2id(
                    context_list[i][split][index], word2id, max_len)
                text_idx = tokens2id(
                    text_list[i][split][index], word2id, max_len)
                idx_data[split].append(ctxt_idx + text_idx)
        idx_data_list.append(idx_data)
    return idx_data_list


def gen_word_vocab(text_data, context_data, k):
    splits = ["train", "valid", "test"]
    ##### Convert tab-separated tweets into list of tokens #####
    text_word_list, ctxt_word_list = [], []
    for i in range(k):
        _text, _ctxt = {}, {}
        for split in splits:
            text_tokens = [tweet.rstrip().split()
                           for tweet in text_data[i][split]]
            ctxt_tokens = [tweet.rstrip().split()
                           for tweet in context_data[i][split]]
            _text[split] = text_tokens
            _ctxt[split] = ctxt_tokens
        text_word_list.append(_text)
        ctxt_word_list.append(_ctxt)

    ##### Create Vocab (id2word, word2id) #####
    vocab = Counter()

    max_len = 0
    for i in range(k):
        for split in splits:
            for tokens in text_word_list[i][split]:
                vocab.update(tokens)
                if max_len < len(tokens):
                    max_len = len(tokens)
            for tokens in ctxt_word_list[i][split]:
                vocab.update(tokens)
                if max_len < len(tokens):
                    max_len = len(tokens)
    # max_len = 100 if max_len > 100 else max_len
    max_len = 100

    count = 0
    for word in vocab.keys():
        if vocab[word] >= 2:  # only count the word with frequencies of at least 2
            count += 1
    vocab_size = count

    _vocab = ["PAD", "UNK"]  # add special vocab for PAD and UNK
    for word, _ in vocab.most_common(vocab_size):
        _vocab.append(word)

    # create dictionaries
    id2word = {}
    word2id = {}
    for i, word in enumerate(_vocab):
        id2word[i] = word
        word2id[word] = i

    # Representing words in tweets into their indices
    ctxt_text_idx = idx_representations_of_text(
        text_word_list, ctxt_word_list, word2id, max_len, k)

    return id2word, word2id, ctxt_text_idx


def gen_word_embeddings(mode: EmbeddingMode, path):
    print(f'Mode: {mode}')

    embedding_dim = 300
    with open(path + "/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab["word2id"].keys())
        print("Vocabulary loaded with %s words" % vocab_size)

    glove_embedding_matrix = np.zeros((vocab_size, embedding_dim))

    if mode == EmbeddingMode.GLOVE:
        glove = {}
        print("Loading pre-trained embedding.")
        f = open(path + "/../../glove.840B.300d.txt")
        for line in f:
            values = line.split()
            word = ' '.join(values[:-embedding_dim])
            coefs = np.asarray(values[-embedding_dim:], dtype='float32')
            glove[word] = coefs

        for word in vocab["word2id"]:
            if word == "PAD":
                glove_embedding_matrix[vocab["word2id"]
                                       [word]] = np.zeros(embedding_dim)
            elif word in glove:
                glove_embedding_matrix[vocab["word2id"][word]] = glove[word]
            else:
                glove_embedding_matrix[vocab["word2id"][word]] = np.random.normal(
                    0, 0.01, embedding_dim)

        np.save(path + "/embedding.npy", glove_embedding_matrix)

    # Bert embeddings
    elif mode == EmbeddingMode.BERT:
        print("Generating bert embedding.")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_seq_length = 128  # Your choice here.
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(
            shape=(
                max_seq_length,
            ),
            dtype=tf.int32,
            name="input_mask")
        segment_ids = tf.keras.layers.Input(
            shape=(
                max_seq_length,
            ),
            dtype=tf.int32,
            name="segment_ids")
        bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True)
        pooled_output, sequence_output = bert_layer(
            [input_word_ids, input_mask, segment_ids])

        model = Model(
            inputs=[
                input_word_ids,
                input_mask,
                segment_ids],
            outputs=[
                pooled_output,
                sequence_output])

        for word in vocab["word2id"]:
            if word == "PAD":
                glove_embedding_matrix[vocab["word2id"]
                                       [word]] = np.zeros(embedding_dim)
            else:
                tokens = f"[CLS]{tokenizer.tokenize(word)}[SEP]"
                input_ids = get_ids(tokens, tokenizer, max_seq_length=128)
                input_masks = get_masks(tokens, max_seq_length=128)
                input_segments = get_segments(tokens, max_seq_length=128)

                pool_embeddings, all_embeddings = model.predict(
                    [[input_ids], [input_masks], [input_segments]])

                embedding = all_embeddings

                glove_embedding_matrix[vocab["word2id"]
                                       [word]] = np.mean(embedding, axis=0)

        np.save(path + "/embedding.npy", glove_embedding_matrix)

    elif mode == EmbeddingMode.OTHER:
        print("Generating random embedding.")
        for word in vocab["word2id"]:
            if word == "PAD":
                glove_embedding_matrix[vocab["word2id"]
                                       [word]] = np.zeros(embedding_dim)
            else:
                glove_embedding_matrix[vocab["word2id"][word]] = np.random.normal(
                    0, 0.01, embedding_dim)
        np.save(path + "/embedding.npy", glove_embedding_matrix)
    else:
        print(f"Embedding mode not found: {mode}")


##########################################################################
############################## Char Vocab Generation #####################
##########################################################################


def char_embedding(text, max_len=140):
    features = list(
        "abcdefghijklmnopqrstuvwxyz0123456789 \u2014,;.!?:\u201c\u201d’/|_@#$%ˆ&*~‘+-=<>()[]{}")
    _text = ""
    for c in list(text):
        if c in features:
            _text += c
    tokens = list(_text)
    vector = -np.ones(max_len)

    for i, t in enumerate(tokens):
        if i < max_len:
            try:
                j = features.index(t)
            except ValueError:
                j = -1
                print("value error")
            vector[i] = j

    return vector


def gen_char_vocab(text_data, context_data, k):
    _text_data, _ctxt_data = [], []
    splits = ["train", "valid", "test"]
    for index in range(k):
        _text = {"train": [], "valid": [], "test": []}
        _ctxt = {"train": [], "valid": [], "test": []}
        _text_data.append(_text)
        _ctxt_data.append(_ctxt)
    for index in range(k):
        for split in splits:
            for text in text_data[index][split]:
                _vector = char_embedding(text)
                _text_data[index][split].append(_vector)
            for context in context_data[index][split]:
                _vector = char_embedding(context)
                _ctxt_data[index][split].append(_vector)

    return _text_data, _ctxt_data


##########################################################################
############################## Generating Numpy of Input #################
##########################################################################


def save_kfold_npy(data_list, name, path, k):
    file_format = path + "/%s/%s_%s.npy"
    for i in range(k):
        fold_path = path + "/" + str(i)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        for key in data_list[i].keys():
            file_name = file_format % (str(i), name, key)
            array = np.array(data_list[i][key])
            np.save(file_name, array)
            # print("Saved in %s. %s" % (file_name, str(array.shape)))


def process_word_vocab(target_path, mode: EmbeddingMode, k):
    print("Processing word vocab....")
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    with open(data_path + "/data_splits.pkl", "rb") as f:
        _data = pickle.load(f)
    _text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
    id2word, word2id, ctxt_text_idx = gen_word_vocab(
        _text_split, _ctxt_split, k)

    save_kfold_npy(ctxt_text_idx, "CtxtText_InputText", target_path, k)
    save_kfold_npy(_label_split, "Label", target_path, k)

    with open(target_path + "/vocab.pkl", "wb") as f:
        pickle.dump({"word2id": word2id, "id2word": id2word}, f)
    print("Done creating vocabulary pickles")

    gen_word_embeddings(mode, target_path)


def process_char_vocab(target_path, k):
    print("Processing char vocab....")
    _data = load_data_splits()
    _text_split, _ctxt_split, _label_split = _data["text_data"], _data["context_data"], _data["label_data"]
    text_data, ctxt_data = gen_char_vocab(_text_split, _ctxt_split, k)
    save_kfold_npy(text_data, "Char_InputText", target_path, k)
    save_kfold_npy(ctxt_data, "Char_CtxtText", target_path, k)
    save_kfold_npy(_label_split, "Label", target_path, k)


def load_data_splits() -> Dict[str, Any]:
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    with open(data_path + "/data_splits.pkl", "rb") as f:
        return pickle.load(f)


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids: Union[List[int],
                     int] = tokenizer.convert_tokens_to_ids(tokens=tokens)
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="glove")
    parser.add_argument("--num_splits", type=int, default=10)
    args = vars(parser.parse_args())
    mode = EmbeddingMode.from_str(args["mode"])

    target_path = os.path.dirname(
        os.path.abspath(__file__)) + "/../data/target"
    process_word_vocab(target_path, mode, args["num_splits"])
    process_char_vocab(target_path, args["num_splits"])
