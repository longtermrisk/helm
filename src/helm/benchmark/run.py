import argparse
import copy
from dataclasses import replace
from typing import List, Optional

from helm.benchmark.presentation.run_entry import RunEntry, read_run_entries
from helm.common.hierarchical_logger import hlog, htrack, htrack_block
from helm.common.authentication import Authentication
from helm.common.object_spec import parse_object_spec
from helm.proxy.clients.huggingface_model_registry import (
    register_huggingface_hub_model_config,
    register_huggingface_local_model_config,
)
from helm.proxy.clients.remote_model_registry import (
    check_and_register_remote_model,
)
from helm.proxy.services.remote_service import (
    create_authentication,
    add_service_args,
)

from helm.benchmark.model_metadata_registry import (
    register_model_metadata_from_path,
)
from helm.benchmark.model_deployment_registry import (
    register_model_deployments_from_path,
)
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from .executor import ExecutionSpec
from .runner import Runner, RunSpec, LATEST_SYMLINK
from .slurm_runner import SlurmRunner
from .run_specs import construct_run_specs
import helm.common.clr_contrib as clr_contrib


def run_entries_to_run_specs(
    run_entries: List[RunEntry],
    max_eval_instances: Optional[int] = None,
    num_train_trials: Optional[int] = None,
    models_to_run: Optional[List[str]] = None,
    groups_to_run: Optional[List[str]] = None,
    priority: Optional[int] = None,
) -> List[RunSpec]:
    """Runs RunSpecs given a list of RunSpec descriptions."""
    run_specs: List[RunSpec] = []
    for entry in run_entries:
        # Filter by priority
        if priority is not None and entry.priority > priority:
            continue

        for run_spec in construct_run_specs(
            parse_object_spec(entry.description)
        ):
            # Filter by models
            if (
                models_to_run
                and run_spec.adapter_spec.model not in models_to_run
            ):
                continue

            # Filter by groups
            if groups_to_run and not any(
                group in groups_to_run for group in run_spec.groups
            ):
                continue

            # Modify AdapterSpec
            adapter_spec: AdapterSpec = run_spec.adapter_spec
            if max_eval_instances is not None:
                adapter_spec = replace(
                    adapter_spec, max_eval_instances=max_eval_instances
                )
            if (
                num_train_trials is not None
                or adapter_spec.max_train_instances == 0
            ):
                adapter_spec = replace(
                    adapter_spec,
                    num_train_trials=1
                    if adapter_spec.max_train_instances == 0
                    else num_train_trials,
                )
            if clr_contrib.USE_SINGLE_STEP_SG_IMPLEMENTATION:
                adapter_spec = replace(
                    adapter_spec,
                    instructions=clr_contrib.SINGLE_STEP_PROMPT
                    + adapter_spec.instructions,
                )
            # Not working, see the clr_contrib.py file for explanations
            # if (
            #     clr_contrib.USE_SINGLE_STEP_SG_IMPLEMENTATION
            #     or clr_contrib.USE_THREE_STEPS_SG_IMPLEMENTATION
            # ):
            #     print("Setting CURRENT_RUN_SPEC_ADAPTER_SPEC")
            #     print("adapter_spec", adapter_spec)
            #     clr_contrib.CURRENT_RUN_SPEC_ADAPTER_SPEC = copy.deepcopy(
            #         adapter_spec
            #     )
            run_spec = replace(run_spec, adapter_spec=adapter_spec)
            # Append groups
            if entry.groups is not None:
                groups_name: str = (
                    ""
                    if len(entry.groups) == 0
                    else f",groups={'-'.join(sorted(entry.groups))}"
                )
                run_spec = replace(
                    run_spec,
                    name=run_spec.name + groups_name,
                    groups=run_spec.groups + entry.groups,
                )

            run_specs.append(run_spec)

    return run_specs


def run_benchmarking(
    run_specs: List[RunSpec],
    auth: Authentication,
    url: Optional[str],
    local_path: str,
    num_threads: int,
    output_path: str,
    suite: str,
    dry_run: bool,
    skip_instances: bool,
    cache_instances: bool,
    cache_instances_only: bool,
    skip_completed_runs: bool,
    exit_on_error: bool,
    use_slurm_runner: bool,
    mongo_uri: str = "",
) -> List[RunSpec]:
    """Runs RunSpecs given a list of RunSpec descriptions."""
    execution_spec = ExecutionSpec(
        auth=auth,
        url=url,
        local_path=local_path,
        parallelism=num_threads,
        dry_run=dry_run,
        mongo_uri=mongo_uri,
    )
    with htrack_block("run_specs"):
        for run_spec in run_specs:
            hlog(run_spec)
    runner_cls = SlurmRunner if use_slurm_runner else Runner
    runner: Runner = runner_cls(
        execution_spec,
        output_path,
        suite,
        skip_instances,
        cache_instances,
        cache_instances_only,
        skip_completed_runs,
        exit_on_error,
    )
    runner.run_all(run_specs)
    return run_specs


def add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Where to save all the output",
        default="benchmark_output",
    )
    parser.add_argument(
        "-n",
        "--num-threads",
        type=int,
        help="Max number of threads to make requests",
        default=4,
    )
    parser.add_argument(
        "--skip-instances",
        action="store_true",
        help="Skip creation of instances (basically do nothing but just parse everything).",
    )
    parser.add_argument(
        "--cache-instances",
        action="store_true",
        help="Save generated instances input to model to disk. If already cached, read instances from file.",
    )
    parser.add_argument(
        "--cache-instances-only",
        action="store_true",
        help="Generate and save instances for scenario ONLY (i.e. do not evaluate models on instances).",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Skip execution, only output scenario states and estimate token usage.",
    )
    parser.add_argument(
        "-m",
        "--max-eval-instances",
        type=int,
        required=True,
        help="Maximum number of instances to evaluate on, overrides the value in Adapter spec.",
    )
    parser.add_argument(
        "-t",
        "--num-train-trials",
        type=int,
        help="Number of trials where each trial samples a different set of in-context examples. "
        "Overrides the value in Adapter spec.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Name of the suite this run belongs to (default is today's date).",
        required=True,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="DEPRECATED: Does nothing. Do not use. Previously enabled local mode. "
        "Now does nothing and will be removed in the next released version. "
        "Local mode is enabled by default, and only disabled if the --server_url flag is set.",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="If running locally, the path for `ServerService`.",
        default="prod_env",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        help="If non-empty, the URL of the MongoDB database that will be used for caching instead of SQLite",
        default="",
    )


def validate_args(args):
    assert (
        args.suite != LATEST_SYMLINK
    ), f"Suite name can't be '{LATEST_SYMLINK}'"
    if args.cache_instances_only:
        assert (
            args.cache_instances
        ), "If --cache-instances-only is set, --cache-instances must also be set."


def get_args_from_parser():
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument(
        "-c",
        "--conf-paths",
        nargs="+",
        help="Where to read RunSpecs to run from",
        default=[],
    )
    parser.add_argument(
        "--models-to-run",
        nargs="+",
        help="Only RunSpecs with these models specified. If no model is specified, runs with all models.",
        default=None,
    )
    parser.add_argument(
        "--groups-to-run",
        nargs="+",
        help="Only RunSpecs with these (scenario) groups specified. "
        "If no group is specified, runs with all groups.",
        default=None,
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Fail and exit immediately if a particular RunSpec fails.",
    )
    parser.add_argument(
        "--skip-completed-runs",
        action="store_true",
        help="Skip RunSpecs that have completed i.e. output files exists.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Run RunSpecs with priority less than or equal to this number. "
        "If a value for --priority is not specified, run on everything",
    )
    parser.add_argument(
        "-r", "--run-specs", nargs="*", help="Specifies what to run", default=[]
    )
    parser.add_argument(
        "--enable-huggingface-models",
        nargs="+",
        default=[],
        help="Experimental: Enable using AutoModelForCausalLM models from Hugging Face Model Hub. "
        "Format: namespace/model_name[@revision]",
    )
    parser.add_argument(
        "--enable-local-huggingface-models",
        nargs="+",
        default=[],
        help="Experimental: Enable using AutoModelForCausalLM models from a local path.",
    )
    parser.add_argument(
        "--enable-remote-models",
        nargs="+",
        default=[],
        help="Experimental: Enable remote service models that are not available on the client. "
        "The client will use RemoteWindowService for windowing.",
    )
    parser.add_argument(
        "--use-slurm-runner",
        action="store_true",
        help="Experimental: If set, each RunSpec will be run in a separate worker Slurm job. "
        "Currently only works on the Stanford NLP cluster.",
    )
    parser.add_argument(
        "--model-metadata-paths",
        nargs="+",
        help="Experimental: Where to read model metadata from",
        default=[],
    )
    parser.add_argument(
        "--model-deployment-paths",
        nargs="+",
        help="Experimental: Where to read model deployments from",
        default=[],
    )
    add_run_args(parser)
    args = parser.parse_args()
    return args


@htrack(None)
def main():
    args = get_args_from_parser()
    validate_args(args)

    for huggingface_model_name in args.enable_huggingface_models:
        register_huggingface_hub_model_config(huggingface_model_name)
    for huggingface_model_path in args.enable_local_huggingface_models:
        register_huggingface_local_model_config(huggingface_model_path)
    for model_metadata_path in args.model_metadata_paths:
        register_model_metadata_from_path(model_metadata_path)
    for model_deployment_paths in args.model_deployment_paths:
        register_model_deployments_from_path(model_deployment_paths)

    if args.server_url and args.enable_remote_models:
        check_and_register_remote_model(
            args.server_url, args.enable_remote_models
        )

    run_entries: List[RunEntry] = []
    if args.conf_paths:
        run_entries.extend(read_run_entries(args.conf_paths).entries)
    if args.run_specs:
        run_entries.extend(
            [
                RunEntry(description=description, priority=1, groups=None)
                for description in args.run_specs
            ]
        )

    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
        max_eval_instances=args.max_eval_instances,
        num_train_trials=args.num_train_trials,
        models_to_run=args.models_to_run,
        groups_to_run=args.groups_to_run,
        priority=args.priority,
    )
    hlog(f"{len(run_entries)} entries produced {len(run_specs)} run specs")

    if len(run_specs) == 0:
        hlog("There were no RunSpecs or they got filtered out.")
        return

    auth: Authentication = (
        Authentication("")
        if args.skip_instances or not args.server_url
        else create_authentication(args)
    )

    run_benchmarking(
        run_specs=run_specs,
        auth=auth,
        url=args.server_url,
        local_path=args.local_path,
        num_threads=args.num_threads,
        output_path=args.output_path,
        suite=args.suite,
        dry_run=args.dry_run,
        skip_instances=args.skip_instances,
        cache_instances=args.cache_instances,
        cache_instances_only=args.cache_instances_only,
        skip_completed_runs=args.skip_completed_runs,
        exit_on_error=args.exit_on_error,
        use_slurm_runner=args.use_slurm_runner,
        mongo_uri=args.mongo_uri,
    )

    if args.local:
        hlog(
            "WARNING: The --local flag is deprecated. It now does nothing and will be removed in "
            "the next released version. Local mode is enabled by default, and only disabled if the "
            "--server_url flag is set. Please remove --local from your command."
        )

    hlog("Done.")


if __name__ == "__main__":
    main()
