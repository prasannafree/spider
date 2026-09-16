from datasets import load_dataset


DATASET_NAME = "xlangai/spider"


def main():
    print("Downloading Spider 1.0...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset:")
    print(dataset)

    print("\nTraining examples:", len(dataset["train"]))
    print("Validation examples:", len(dataset["validation"]))

    print("\nFirst training example:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
    