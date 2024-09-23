import sys

from prefixspan import prefixspan

from experiments.log_analysis.deepspan_deinterleave.datasets.real import make_dataset

NUM_STATES = 8
NUM_CHAINS = 3
LEN_WINDOW = "5ms"
LEN_DATASET_MAX = 1000
MINSUP = 30
LEN_SEQUENCE_MIN = 2
LEN_SEQUENCE_MAX = 16


SUBJECTS = ["org.apache.zookeeper"]


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def main():
    for subject in SUBJECTS:
        dataset = make_dataset(
            subject,
            window_size=LEN_WINDOW,
            max_sequence_length=LEN_SEQUENCE_MAX,
            max_dataset_length=LEN_DATASET_MAX,
        )
        trie = prefixspan(dataset, MINSUP)
        sys.stdout.write(repr(trie) + "\n")
        break
    return None


if __name__ == "__main__":
    main()
