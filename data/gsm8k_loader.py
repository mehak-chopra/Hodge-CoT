from datasets import load_dataset


class GSM8KLoader:
    """
    Loader and interface for the GSM8K dataset.

    GSM8K provides two official splits:
        - train
        - test

    A validation subset can optionally be created from the
    training split for model development and experimentation.
    """

    def __init__(self, validation_size=0.1, seed=42):
        """
        Initialize the GSM8K loader.

        Parameters
        ----------
        validation_size : float
            Fraction of the training data to use for validation.

        seed : int
            Random seed used when creating the validation split.
        """

        if not 0 <= validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        self.validation_size = validation_size
        self.seed = seed

        self.dataset = None
        self.train = None
        self.validation = None
        self.test = None

    def load(self):
        """
        Load GSM8K and prepare train, validation, and test splits.

        Returns
        -------
        dict
            Dictionary containing train, validation, and test datasets.
        """

        print("Loading GSM8K...")

        self.dataset = load_dataset(
            "openai/gsm8k",
            "main"
        )

        # Official GSM8K training split
        training_data = self.dataset["train"]

        # Create validation subset from training data.
        split_data = training_data.train_test_split(
            test_size=self.validation_size,
            seed=self.seed
        )

        self.train = split_data["train"]
        self.validation = split_data["test"]

        # Official GSM8K test split
        self.test = self.dataset["test"]

        print("\nGSM8K loaded successfully!")

        print(f"Training examples:   {len(self.train)}")
        print(f"Validation examples: {len(self.validation)}")
        print(f"Test examples:       {len(self.test)}")

        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test
        }

    def get_split(self, split):
        """
        Return a requested dataset split.

        Parameters
        ----------
        split : str
            One of:
                "train"
                "validation"
                "test"

        Returns
        -------
        Dataset
            Requested dataset split.
        """

        if self.dataset is None:
            self.load()

        if split == "train":
            return self.train

        if split == "validation":
            return self.validation

        if split == "test":
            return self.test

        raise ValueError(
            "Invalid split. Choose 'train', 'validation', or 'test'."
        )

    def get_example(self, split, index=0):
        """
        Return one example from a dataset split.

        Parameters
        ----------
        split : str
            Dataset split.

        index : int
            Index of the example.

        Returns
        -------
        dict
            GSM8K example containing question and answer.
        """

        dataset_split = self.get_split(split)

        return dataset_split[index]

    def show_example(self, split="train", index=0):
        """
        Display one GSM8K example.
        """

        example = self.get_example(split, index)

        print("\nQuestion:")
        print(example["question"])

        print("\nAnswer and Solution:")
        print(example["answer"])


if __name__ == "__main__":

    loader = GSM8KLoader(
        validation_size=0.1,
        seed=42
    )

    loader.load()

    print("\nExample from training split:")
    loader.show_example("train", 0)

    print("\nExample from validation split:")
    loader.show_example("validation", 0)

    print("\nExample from test split:")
    loader.show_example("test", 0)