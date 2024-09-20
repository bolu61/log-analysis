import sys

from experiments.log_analysis.deepspan_deinterleave.datasets.real import make_database
from prefixspan import prefixspan

NUM_STATES = 8
NUM_CHAINS = 3
LEN_SEQUENCE = 24
LEN_DATASET = 10_000
MINSUP = 2_000


SUBJECTS = ["org.apache.zookeeper"]


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def main():
    for subject in SUBJECTS:
        dataset = make_database(subject, window_size="8s", min_length=2)

        trie = prefixspan([*dataset], minsup=len(dataset) // 5)  # fmt: ignore
        sys.stdout.write(str(trie) + "\n")
        return trie
    return None

    # trie = prefixspan([*map(lambda a: a.tolist(), database)], minsup=int(len(database) * 0.2))

    # groups = []
    # for group in separate(
    #     trie, df.to_records(), maxlen=24, key=lambda x: as_id(x.EventId)
    # ):
    #     groups.append(jnp.array([as_id(g.EventId) for g in group]))

    # print(f"grouping_length: {grouping_length(groups)}")


if __name__ == "__main__":
    main()
