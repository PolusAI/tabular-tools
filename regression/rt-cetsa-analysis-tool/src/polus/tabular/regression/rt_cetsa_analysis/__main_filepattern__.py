"""Main with filepattern support."""
# """CLI for rt-cetsa-moltprot-tool."""


# # get env

# # Initialize the logger
# logging.basicConfig(


# @app.command()
# def main(
#     inp_dir: pathlib.Path = typer.Option(
#         ...,
#         "--inpDir",
#     ),
#     params_pattern: str = typer.Option(
#         ".+",
#         "--params",
#     ),
#     values_pattern: str = typer.Option(
#         ".+",
#         "--values",
#     ),
#     platemap_pattern: str = typer.Option(
#         ".+",
#         "--platemap",
#     ),
#     preview: bool = typer.Option(
#         False,
#         "--preview",
#     ),
#     out_dir: pathlib.Path = typer.Option(
#         ...,
#         "--outDir",
#     ),
# ) -> None:
#     """CLI for rt-cetsa-moltprot-tool."""
#     # TODO: Add to docs that input csv file should be sorted by `Temperature` column.


#     if preview:
#         with (out_dir / "preview.json").open("w") as f:

#     for params, values, platemap in zip(params_files, values_files, platemap_files):
#         # TODO replace with exceptions
#         if len(params[1]) != 1 or len(values[1]) != 1 or len(platemap[1]) != 1:
#             raise Exception(
#                 msg,


# if __name__ == "__main__":
