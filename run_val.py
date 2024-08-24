import click

from pipelines import leave_one_group_out_pipeline
from utils.definitions import ROOT_DIR
from utils.generic_helper import load_yaml_file


@click.command(
    help="""
    Entry point for running leave-one-group-out
    cross-validation pipeline. 
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
    "--model-type",
    default="eol",
    help="""State which model type to run.
        Valid options are 'eol', 'rul',
        or 'classification'. Default to 'eol'.
        """,
)
def main(
    not_loaded: bool = False,
    model_type: str = "eol",
) -> None:
    MODEL_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/config/model_config.yaml")

    leave_one_group_out_pipeline(
        not_loaded=not_loaded,
        model_type=model_type,
        threshold=MODEL_CONFIG["threshold"],
        signature_depth=MODEL_CONFIG[model_type]["signature_depth"],
        param_grid=MODEL_CONFIG[model_type]["param_grid"],
        scorer=MODEL_CONFIG[model_type]["scorer"],
    )


if __name__ == "__main__":
    main()
