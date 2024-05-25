import click

from pipelines import training_pipeline
from utils.generic_helper import load_yaml_file
from utils.definitions import ROOT_DIR


@click.command(
    help="""
    Entry point for running eol, rul,
    and classification models.
    """
)
@click.option(
    "--not-loaded",
    is_flag=True,
    default=False,
    help="""If given, raw data will be loaded and 
    cleaned.
        """,
)
@click.option(
    "--no-proposed-split",
    is_flag=True,
    default=False,
    help="""If given, train-test split used 
    in this study will not be used for modelling.
        """,
)
@click.option(
    "--model-type",
    default="eol",
    help="""State which model type to run.
        Valid options are 'eol', 'rul',
        or 'classification'. Default to 'eol'.
        """,
)
def main(
    not_loaded: bool = False,
    no_proposed_split: bool = False,
    model_type: str = "eol",
) -> None:
    MODEL_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/config/model_config.yaml")

    training_pipeline(
        not_loaded=not_loaded,
        no_proposed_split=no_proposed_split,
        test_size=MODEL_CONFIG["test_size"],
        model_type=model_type,
        signature_depth=MODEL_CONFIG[model_type]["signature_depth"],
        threshold=MODEL_CONFIG["threshold"],
        param_grid=MODEL_CONFIG[model_type]["param_grid"],
        scorer=MODEL_CONFIG[model_type]["scorer"],
    )


if __name__ == "__main__":
    main()
