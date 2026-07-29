"""Command line interface for DeepBioisostere.

Heavy imports (torch, rdkit, the model) are deferred into the command bodies so
that ``deepbioisostere --help`` stays fast and so that ``--help`` works even in
an environment where the scientific stack is broken.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

# The package-wide logger the commands configure. `assets` logs onto a child of
# it, and Generator is handed its .info so that one switch covers both.
_asset_logger = logging.getLogger("deepbioisostere")


def _setup_logging(quiet: bool) -> None:
    """Send package progress messages to stderr, unless suppressed.

    On by default: the first run of ``generate`` spends minutes downloading a
    712 MB tensor cache, and silence during that is indistinguishable from a
    hang. stderr rather than stdout so that piping the csv stays clean.

    Configuring the root logger is an application's job, not a library's, which
    is why this lives here and ``assets`` only ever calls ``logger.info``.
    """
    if quiet:
        # A NullHandler rather than a raised level, so nothing reaches the
        # "no handlers could be found" fallback either.
        _asset_logger.addHandler(logging.NullHandler())
        _asset_logger.propagate = False
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _asset_logger.addHandler(handler)
    _asset_logger.setLevel(logging.INFO)
    _asset_logger.propagate = False


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Autonomous bioisosteric replacement for multi-property optimization.",
    no_args_is_help=True,
)

fragment_library_app = typer.Typer(
    add_completion=False,
    help="Fragment library preparation.",
    no_args_is_help=True,
)
app.add_typer(fragment_library_app, name="fragment-library")


def _fail(message: str) -> "typer.Exit":
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _parse_targets(targets: list[str]) -> dict[str, float]:
    """Parse repeated ``NAME=VALUE`` options into a property -> target mapping."""
    parsed: dict[str, float] = {}
    for item in targets:
        if "=" not in item:
            raise _fail(
                f"Invalid --target {item!r}. Expected NAME=VALUE, e.g. 'logp=-1'."
            )
        name, _, raw = item.partition("=")
        name = name.strip().lower()
        try:
            parsed[name] = float(raw)
        except ValueError:
            raise _fail(
                f"Invalid --target {item!r}: {raw!r} is not a number."
            ) from None
    return parsed


def _read_smiles_file(path: Path) -> list[str]:
    if not path.is_file():
        raise _fail(f"Input file not found: {path}")
    smiles: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept a bare SMILES list or the first column of a csv/tsv.
        smiles.append(line.replace("\t", ",").split(",")[0].strip())
    return smiles


@app.command()
def generate(
    smiles: list[str] = typer.Option(
        [], "--smiles", "-s", help="Input SMILES. Repeatable."
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="File of SMILES, one per line (or first csv column).",
    ),
    target: list[str] = typer.Option(
        [],
        "--target",
        "-t",
        help=(
            "Target change for a property as NAME=VALUE, e.g. -t mw=0 -t logp=-1. "
            "Repeatable. The set of names selects the checkpoint."
        ),
    ),
    output: Path = typer.Option(
        Path("generation_result.csv"), "--output", "-o", help="Output csv path."
    ),
    num_samples: int = typer.Option(
        100, "--num-samples", "-n", help="Samples generated per input molecule."
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda:0."),
    batch_size: int = typer.Option(512, help="Batch size."),
    num_cores: int = typer.Option(4, help="Dataloader worker processes."),
    new_frag_type: str = typer.Option(
        "all", help="Insertion fragment split: train, val, test or all."
    ),
    ablation: bool = typer.Option(
        False, "--ablation", help="Use the use_subgraph_AMPN=False variant."
    ),
    model_dir: Optional[Path] = typer.Option(
        None, help="Directory of .pt checkpoints. Defaults to cache/Hub."
    ),
    frag_lib_dir: Optional[Path] = typer.Option(
        None, help="Fragment library directory. Defaults to cache/Hub."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed."),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress messages about asset resolution and download.",
    ),
) -> None:
    """Generate bioisosteric replacements for one or more molecules.

    Example:

        deepbioisostere generate -s "Cc1ccc2cnc(N(C)CCc3ccccn3)nc2c1" -t mw=0 -t logp=-1
    """
    _setup_logging(quiet)
    inputs = list(smiles)
    if input_file is not None:
        inputs.extend(_read_smiles_file(input_file))
    if not inputs:
        raise _fail("No input molecules. Pass --smiles/-s or --input/-i.")
    if not target:
        raise _fail(
            "No targets. Pass at least one --target/-t, e.g. -t mw=0 -t logp=-1."
        )

    targets = _parse_targets(target)
    properties = sorted(targets)

    from .assets import AssetError
    from .conditioning import Conditioner
    from .generate import Generator
    from .model import DeepBioisostere
    from .utils import set_seed

    if seed is not None:
        set_seed(seed)

    try:
        model = DeepBioisostere.from_pretrained(
            properties=properties, ablation=ablation, local_dir=model_dir
        )
    except AssetError as exc:
        raise _fail(str(exc)) from None

    conditioner = Conditioner(phase="generation", properties=properties)
    try:
        generator = Generator(
            model=model,
            processed_frag_dir=frag_lib_dir,
            conditioner=conditioner,
            device=device,
            num_cores=num_cores,
            batch_size=batch_size,
            new_frag_type=new_frag_type,
            num_sample_each_mol=num_samples,
            properties=properties,
            # Generator defaults to bare print(); route it through the same
            # switch so --quiet silences everything the package emits, not
            # just the asset messages.
            logger=(lambda *_a, **_k: None) if quiet else _asset_logger.info,
        )
    except AssetError as exc:
        raise _fail(str(exc)) from None

    result = generator.generate([(smi, dict(targets)) for smi in inputs])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    typer.secho(
        f"Wrote {len(result)} generated molecules to {output}",
        fg=typer.colors.GREEN,
    )


@app.command()
def download(
    target: list[str] = typer.Option(
        [],
        "--properties",
        "-p",
        help="Property set to fetch, comma separated, e.g. -p mw,logp. Repeatable.",
    ),
    all_checkpoints: bool = typer.Option(
        False, "--all", help="Fetch every published checkpoint."
    ),
    fragment_library: bool = typer.Option(
        True, help="Also fetch the fragment library."
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress messages about asset resolution and download.",
    ),
) -> None:
    """Pre-fetch checkpoints and the fragment library into the local cache.

    Useful before going offline, or on a compute node with no outbound network.
    """
    _setup_logging(quiet)
    from .assets import (
        AVAILABLE_ABLATION_SETS,
        AVAILABLE_PROPERTY_SETS,
        AssetError,
        resolve_checkpoint,
        resolve_fragment_library,
    )

    wanted: list[tuple[tuple[str, ...], bool]] = []
    if all_checkpoints:
        wanted += [(s, False) for s in AVAILABLE_PROPERTY_SETS]
        wanted += [(s, True) for s in AVAILABLE_ABLATION_SETS]
    for spec in target:
        wanted.append(
            (tuple(sorted(p.strip().lower() for p in spec.split(","))), False)
        )

    if not wanted and not fragment_library:
        raise _fail("Nothing to do. Pass --properties, --all, or --fragment-library.")

    for properties, is_ablation in wanted:
        label = ",".join(properties) + (" (ablation)" if is_ablation else "")
        try:
            path = resolve_checkpoint(properties, ablation=is_ablation)
        except AssetError as exc:
            raise _fail(str(exc)) from None
        typer.echo(f"  {label}: {path}")

    if fragment_library:
        try:
            path = resolve_fragment_library()
        except AssetError as exc:
            raise _fail(str(exc)) from None
        typer.echo(f"  fragment library: {path}")

    typer.secho("Done.", fg=typer.colors.GREEN)


@app.command()
def train(
    data_path: Path = typer.Option(
        ..., "--data-path", help="Path to the preprocessed training csv."
    ),
    save_name: str = typer.Option(
        ..., "--save-name", help="Run name; output goes to <project-dir>/model_save/."
    ),
    project_dir: Path = typer.Option(
        Path.cwd(), "--project-dir", help="Directory holding model_save/."
    ),
    properties: Optional[str] = typer.Option(
        None,
        "--properties",
        "-p",
        help="Comma separated properties to condition on, e.g. -p mw,logp. "
        "Omit for an unconditioned model.",
    ),
    seed: int = typer.Option(1024, help="Random seed."),
    use_cuda: bool = typer.Option(True, help="Train on GPU."),
    batch_size: int = typer.Option(512, help="Batch size."),
    frag_lib_batch_size: int = typer.Option(512, help="Fragment embedding batch size."),
    num_cores: int = typer.Option(4, help="Dataloader worker processes."),
    lr: float = typer.Option(2e-4, help="Learning rate."),
    lr_reduce_factor: float = typer.Option(0.5, help="ReduceLROnPlateau factor."),
    min_lr: float = typer.Option(1e-7, help="Learning rate floor."),
    patience: int = typer.Option(10, help="Scheduler patience in epochs."),
    threshold: float = typer.Option(1e-3, help="Scheduler improvement threshold."),
    max_epoch: int = typer.Option(1000, help="Maximum epochs."),
    num_neg_sample: int = typer.Option(20, help="Negative samples per datapoint."),
    alpha1: float = typer.Option(0.5, help="Negative-sampling exponent."),
    alpha2: float = typer.Option(0.5, help="Weighted sampler exponent."),
    num_batch_each_epoch: Optional[int] = typer.Option(100, help="Batches per epoch."),
    dropout: float = typer.Option(0.2, help="Dropout probability."),
    use_subgraph_AMPN: bool = typer.Option(
        True,
        "--use-subgraph-ampn/--no-use-subgraph-ampn",
        help="Fragment-level AMPN. Disable for the ablation model.",
    ),
    use_delta: bool = typer.Option(True, help="Condition on property deltas."),
    use_soft_one_hot: bool = typer.Option(False, help="Soft one-hot conditioning."),
    weighted_sampler: bool = typer.Option(True, help="Weight by fragment frequency."),
    lr_scheduler_can_terminate: bool = typer.Option(
        True, help="Let the scheduler end training at min_lr."
    ),
    print_loss: bool = typer.Option(True, help="Log losses as well as probabilities."),
    profiling: bool = typer.Option(False, help="Enable the torch profiler."),
) -> None:
    """Train a model.

    Defaults reproduce the configuration used for the published models
    (see jobscripts/submit_train.sh), so in practice only --data-path,
    --save-name and --properties need to be supplied.
    """
    import argparse as _argparse

    from .model import DeepBioisostere
    from .training import main as train_main

    if not data_path.is_file():
        raise _fail(f"Training data not found: {data_path}")

    prop_list = (
        sorted(p.strip().lower() for p in properties.split(",") if p.strip())
        if properties
        else []
    )

    args = _argparse.Namespace()
    # Architecture defaults live on the model, so a checkpoint stays loadable
    # without repeating them here.
    for key, value in DeepBioisostere.default_args.items():
        setattr(args, key, value)

    args.project_dir = str(project_dir)
    args.save_name = save_name
    args.data_path = data_path
    args.data_dir = str(data_path.parent)
    args.conditioning = bool(prop_list)
    args.properties = prop_list
    args.use_delta = use_delta
    args.use_soft_one_hot = use_soft_one_hot
    args.seed = seed
    args.use_cuda = use_cuda
    args.ngpu = 1
    args.batch_size = batch_size
    args.frag_lib_batch_size = frag_lib_batch_size
    args.num_cores = num_cores
    args.lr = lr
    args.lr_reduce_factor = lr_reduce_factor
    args.min_lr = min_lr
    args.patience = patience
    args.threshold = threshold
    args.max_epoch = max_epoch
    args.num_neg_sample = num_neg_sample
    args.alpha1 = alpha1
    args.alpha2 = alpha2
    args.weighted_sampler = weighted_sampler
    args.lr_scheduler_can_terminate = lr_scheduler_can_terminate
    args.num_batch_each_epoch = num_batch_each_epoch
    args.print_loss = print_loss
    args.profiling = profiling
    args.dropout = dropout
    args.use_subgraph_AMPN = use_subgraph_AMPN

    # train_main() does its own path setup, seeding and logging. Do not
    # duplicate them here: train_path_setting() allocates a fresh numbered run
    # directory on every call, so calling it twice would create two.
    train_main(args)


@app.command()
def info() -> None:
    """Show resolved asset locations and the published checkpoint sets."""
    from . import __version__
    from .assets import (
        AVAILABLE_ABLATION_SETS,
        AVAILABLE_PROPERTY_SETS,
        default_cache_dir,
        hf_repo_id,
    )

    typer.echo(f"deepbioisostere {__version__}")
    typer.echo(f"python           {sys.version.split()[0]}")
    typer.echo(f"cache directory  {default_cache_dir()}")
    typer.echo(f"hugging face     {hf_repo_id()}")
    typer.echo("")
    typer.echo("checkpoints:")
    for combo in AVAILABLE_PROPERTY_SETS:
        typer.echo(f"  {','.join(combo)}")
    typer.echo("ablation checkpoints:")
    for combo in AVAILABLE_ABLATION_SETS:
        typer.echo(f"  {','.join(combo)}")


@fragment_library_app.command("prepare")
def prepare_fragment_library(
    directory: Optional[Path] = typer.Argument(
        None, help="Directory containing fragment_library.csv. Defaults to cache/Hub."
    ),
    num_cores: int = typer.Option(
        4, help="Worker processes. Bounded by default; this is CPU-heavy."
    ),
) -> None:
    """Build the tensor caches from ``fragment_library.csv``.

    This happens automatically on first generation; run it ahead of time to pay
    the cost explicitly, for instance inside a batch job rather than during an
    interactive session.

    ``num_cores`` defaults to 4 rather than to every available core: this
    parses ~291k fragments across a process pool, and on a shared machine
    grabbing every core is antisocial. Raise it to the size of your allocation
    inside a batch job.
    """
    from .assets import AssetError, resolve_fragment_library
    from .fragment_library.parse_fragments import FragLibProcessor

    try:
        frag_dir = (
            Path(directory) if directory is not None else resolve_fragment_library()
        )
    except AssetError as exc:
        raise _fail(str(exc)) from None

    if not (frag_dir / "fragment_library.csv").is_file():
        raise _fail(f"fragment_library.csv not found in {frag_dir}")

    FragLibProcessor.process_frag_library(frag_lib_dir=frag_dir, num_cores=num_cores)
    typer.secho(f"Fragment library prepared in {frag_dir}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
