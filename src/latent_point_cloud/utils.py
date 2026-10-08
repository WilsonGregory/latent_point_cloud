import argparse


def get_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument_group("data", "arguments to specify the data")
    parser.add_argument(
        "--data-dir", help="the directory where the data lives", type=str, default="./data/"
    )
    parser.add_argument("--n-train", help="number of training points", type=int, default=1)
    parser.add_argument("--n-val", help="number of validation points", type=int, default=None)
    parser.add_argument("--n-test", help="number of testing points", type=int, default=None)
    parser.add_argument(
        "--normalize",
        help="normalize input data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument_group("training", "arguments related to training the models")
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=8)
    parser.add_argument("-t", "--n-trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("--seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-v", "--verbose", help="verbose argument passed to trainer", type=int, default=1
    )

    # preferred method to save/load the models.
    parser.add_argument_group(
        "saving", "arguments related to saving and loading the models, images"
    )
    parser.add_argument(
        "--model-dir",
        help="directory to store saved models, if this is included they will be saved/loaded",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--overwrite-save-model",
        help="if true, always save the model, otherwise load it if its there. Default False",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--image-dir", help="directory to save images, or None to not save", type=str, default=None
    )

    # wandb arguments
    parser.add_argument(
        "--wandb",
        help="whether to use wandb on this run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-entity", help="the wandb user", type=str)

    return parser
