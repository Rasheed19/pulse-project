import click

from pipelines import plot_pipeline


@click.command(
    help="""
    Entry point for plots from various experiments
    carried out in this research.
    """
)
def main() -> None:

    plot_pipeline()


if __name__ == "__main__":
    main()
