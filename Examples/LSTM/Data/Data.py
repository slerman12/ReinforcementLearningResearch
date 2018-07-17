import random
import re
import collections
import numpy as np


class ReadFables:
    def __init__(self, filename):
        # Fables
        self.fables = []

        # Dictionaries for embeddings words to numbers
        self.dictionary = {}
        self.reverse_dictionary = {}

        # For keeping track of batches
        self.batch_begin = 0

        # Fables
        with open(filename) as file:
            fables = (fable.rstrip() for fable in file)
            fables = [fable for fable in fables if fable]
        fables = fables[1::2]

        # Fables split by words and fable
        split_fables = [fables[fable].split() for fable in range(len(fables))]

        # Max fable length
        self.max_fable_length = len(max(split_fables, key=lambda fable: len(fable))) - 1

        # All words
        all_words = [re.sub(r'[^\w\s]', '', word) for fable in split_fables for word in fable]

        # Get most common words
        most_common = collections.Counter(all_words).most_common()

        # Create dictionaries
        for word, _ in most_common:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        # Word dim
        self.word_dim = len(self.dictionary)

        # Encoded fables + padding
        for fable in split_fables:
            # Encoded words in fable
            words = [self.encode_word(re.sub(r'[^\w]', '', word)) for word in fable]

            # Length
            length = len(words) - 1

            # Inputs + padding TODO: Pad in batch iterator with random buckets
            inputs = np.zeros((self.max_fable_length, self.word_dim), dtype=np.int32)
            inputs[:length] = words[:-1]

            # Desired outputs + padding
            desired_outputs = np.zeros((self.max_fable_length, self.word_dim), dtype=np.int32)
            desired_outputs[:length] = words[1:]
            # Fables
            self.fables.append({"inputs": inputs, "desired_outputs": desired_outputs, "length": length})

    def encode_word(self, word):
        # Return encoding
        encoding = np.zeros(self.word_dim)
        encoding[self.dictionary[word]] = 1
        return encoding

    def iterate_batch(self, batch_size):
        # Reset and shuffle batch when all items have been iterated
        if self.batch_begin > len(self.fables) - batch_size:
            # Reset batch index
            self.batch_begin = 0

            # Shuffle fables
            random.shuffle(self.fables)

        # Index of the end boundary of this batch
        batch_end = min(self.batch_begin + batch_size, len(self.fables))

        # Batch
        batch = self.fables[self.batch_begin:batch_end]

        # Update batch index
        self.batch_begin = batch_end

        # Return inputs, desired outputs, and lengths
        return np.stack([fable["inputs"] for fable in batch]), \
            np.stack([fable["desired_outputs"] for fable in batch]), \
            np.array([fable["length"] for fable in batch])


class ToySequenceData(object):
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def iterate_batch(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
