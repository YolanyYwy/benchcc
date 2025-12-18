import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import get_options, run_domain
from tau2.scripts.leaderboard.verify_trajectories import VerificationMode


def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domains,
        help="The domain to run the simulation on",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use for the agent. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"The LLM to use for the user. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on. If not provided, will load 'base' split.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>.json. If not provided, will save to <domain>_<agent>_<user>_<llm_agent>_<llm_user>_<timestamp>.json. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )


def main():
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)
    run_parser.set_defaults(
        func=lambda args: run_domain(
            RunConfig(
                domain=args.domain,
                task_set_name=args.task_set_name,
                task_split_name=args.task_split_name,
                task_ids=args.task_ids,
                num_tasks=args.num_tasks,
                agent=args.agent,
                llm_agent=args.agent_llm,
                llm_args_agent=args.agent_llm_args,
                user=args.user,
                llm_user=args.user_llm,
                llm_args_user=args.user_llm_args,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                save_to=args.save_to,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                log_level=args.log_level,
                enforce_communication_protocol=args.enforce_communication_protocol,
            )
        )
    )

    # Play command
    play_parser = subparsers.add_parser(
        "play", help="Play manual mode - interact with a domain as the agent"
    )
    play_parser.set_defaults(func=lambda args: run_manual_mode())

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation files. Defaults to data/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline', 'mock')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Start command
    start_parser = subparsers.add_parser("start", help="Start all servers")
    start_parser.set_defaults(func=lambda args: run_start_servers())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Submit command with subcommands
    submit_parser = subparsers.add_parser(
        "submit", help="Submission management for the leaderboard"
    )
    submit_subparsers = submit_parser.add_subparsers(
        dest="submit_command", help="Submit subcommands", required=True
    )

    # Submit prepare subcommand
    submit_prepare_parser = submit_subparsers.add_parser(
        "prepare", help="Prepare a submission for the leaderboard"
    )
    submit_prepare_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_prepare_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to save the submission and trajectories",
    )
    submit_prepare_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trajectory verification step",
    )
    submit_prepare_parser.set_defaults(func=lambda args: run_prepare_submission(args))

    # Submit validate subcommand
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate an existing submission directory"
    )
    submit_validate_parser.add_argument(
        "submission_dir",
        help="Path to the submission directory to validate",
    )
    submit_validate_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_validate_parser.set_defaults(func=lambda args: run_validate_submission(args))

    # Submit verify-trajs subcommand
    submit_verify_parser = submit_subparsers.add_parser(
        "verify-trajs", help="Verify trajectory files"
    )
    submit_verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_verify_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_verify_parser.set_defaults(func=lambda args: run_verify_trajectories(args))

    # Add continual learning commands
    add_cl_commands(subparsers)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_start_servers():
    from tau2.scripts.start_servers import main as start_main

    start_main()


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_verify_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.leaderboard.verify_trajectories import verify_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    verify_trajectories(args.paths, args.mode)


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


def run_prepare_submission(args):
    """Run the prepare submission command."""
    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    prepare_submission(
        input_paths=args.input_paths,
        output_dir=args.output,
        run_verification=not args.no_verify,
    )


def run_validate_submission(args):
    """Run the validate submission command."""
    from tau2.scripts.leaderboard.prepare_submission import validate_submission

    validate_submission(submission_dir=args.submission_dir, mode=args.mode)


def run_manual_mode():
    from tau2.scripts.manual_mode import main as manual_main

    manual_main()


def add_cl_run_args(parser):
    """Add continual learning run arguments to a parser."""
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file for CL experiment",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cl_experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["airline", "retail"],
        help="Domains to include in the experiment",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=["sequential", "interleaved", "difficulty"],
        default="sequential",
        help="Curriculum strategy",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["icl_er", "prompt_strategy", "baseline"],
        default="icl_er",
        help="Type of CL agent",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default="gpt-4",
        help="LLM to use for the agent",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-4",
        help="LLM to use for the user simulator",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum few-shot examples in prompt",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Memory buffer size",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks per domain (None = all)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Evaluate every N tasks",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )


def run_cl_experiment_cmd(args):
    """Run continual learning experiment from CLI."""
    from tau2.continual_learning.orchestrator import (
        CLExperimentConfig,
        CLOrchestrator,
        AgentType,
    )
    from tau2.continual_learning.curriculum import CurriculumStrategy
    from tau2.continual_learning.memory import SamplingStrategy

    # Load config from file if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = CLExperimentConfig(**config_dict)
    else:
        # Build config from CLI args
        curriculum_map = {
            "sequential": CurriculumStrategy.SEQUENTIAL,
            "interleaved": CurriculumStrategy.INTERLEAVED,
            "difficulty": CurriculumStrategy.DIFFICULTY_BASED,
        }
        agent_map = {
            "icl_er": AgentType.ICL_ER,
            "prompt_strategy": AgentType.PROMPT_STRATEGY,
            "baseline": AgentType.BASELINE,
        }

        config = CLExperimentConfig(
            name=args.name,
            seed=args.seed,
            output_dir=args.output_dir,
            curriculum_strategy=curriculum_map[args.curriculum],
            domains=args.domains,
            num_tasks_per_domain=args.num_tasks,
            agent_type=agent_map[args.agent_type],
            agent_llm=args.agent_llm,
            max_examples_in_prompt=args.max_examples,
            memory_buffer_size=args.buffer_size,
            eval_frequency=args.eval_frequency,
            user_llm=args.user_llm,
        )

    # Run experiment
    orchestrator = CLOrchestrator(config)
    result = orchestrator.run()

    print(f"\nExperiment complete!")
    print(f"Average Accuracy: {result.metrics.average_accuracy:.3f}")
    print(f"Forgetting Rate: {result.metrics.forgetting_rate:.3f}")
    print(f"Results saved to: {config.output_dir}/{config.name}")


def run_cl_analyze_cmd(args):
    """Analyze CL experiment results."""
    import json
    import os
    from pathlib import Path

    results_path = args.results_file

    # If the path doesn't exist, try to find results.json in subdirectories
    if not os.path.exists(results_path):
        path_obj = Path(results_path)

        # Check if it's a directory path without results.json
        if path_obj.parent.exists() and path_obj.name == "results.json":
            parent_dir = path_obj.parent

            # Find all results.json files in subdirectories
            found_results = list(parent_dir.rglob("results.json"))

            if found_results:
                # Sort by modification time, get the most recent
                found_results.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                results_path = str(found_results[0])
                print(f"Found results file: {results_path}")
            else:
                print(f"Results file not found: {args.results_file}")
                print(f"No results.json found in subdirectories of: {parent_dir}")
                return
        else:
            print(f"Results file not found: {results_path}")
            return

    with open(results_path, 'r') as f:
        results = json.load(f)

    print("=" * 60)
    print("Continual Learning Experiment Analysis")
    print("=" * 60)
    print(f"Experiment: {results['config']['name']}")
    print(f"Domains: {results['config']['domains']}")
    print(f"Curriculum: {results['config']['curriculum_strategy']}")
    print(f"Agent Type: {results['config']['agent_type']}")
    print()
    print("Metrics:")
    print(f"  Average Accuracy: {results['metrics']['average_accuracy']:.3f}")
    print(f"  Forgetting Rate: {results['metrics']['forgetting_rate']:.3f}")
    print(f"  Forward Transfer: {results['metrics']['forward_transfer']:.3f}")
    print(f"  Backward Transfer: {results['metrics']['backward_transfer']:.3f}")
    print()
    print("Per-Domain Accuracy:")
    for domain, acc in results['metrics']['domain_accuracies'].items():
        print(f"  {domain}: {acc:.3f}")
    print()
    print("Per-Domain Forgetting:")
    for domain, fgt in results['metrics']['domain_forgetting'].items():
        print(f"  {domain}: {fgt:.3f}")
    print("=" * 60)


def add_cl_commands(subparsers):
    """Add continual learning commands to the CLI."""

    # CL run command
    cl_run_parser = subparsers.add_parser(
        "cl-run",
        help="Run a continual learning experiment"
    )
    add_cl_run_args(cl_run_parser)
    cl_run_parser.set_defaults(func=lambda args: run_cl_experiment_cmd(args))

    # CL analyze command
    cl_analyze_parser = subparsers.add_parser(
        "cl-analyze",
        help="Analyze continual learning experiment results"
    )
    cl_analyze_parser.add_argument(
        "results_file",
        type=str,
        help="Path to results.json file from CL experiment"
    )
    cl_analyze_parser.set_defaults(func=lambda args: run_cl_analyze_cmd(args))

    # CL info command
    cl_info_parser = subparsers.add_parser(
        "cl-info",
        help="Show information about continual learning module"
    )
    cl_info_parser.set_defaults(func=lambda args: print_cl_info())

    # CL validate-data command
    cl_validate_parser = subparsers.add_parser(
        "cl-validate-data",
        help="Validate task data for CL experiments"
    )
    cl_validate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to tasks.json files or directories"
    )
    cl_validate_parser.add_argument(
        "--domain",
        type=str,
        help="Expected domain name (auto-detected if not provided)"
    )
    cl_validate_parser.set_defaults(func=lambda args: run_cl_validate_data(args))

    # CL data-requirements command
    cl_requirements_parser = subparsers.add_parser(
        "cl-data-requirements",
        help="Analyze data requirements for CL experiments"
    )
    cl_requirements_parser.add_argument(
        "--domains",
        nargs="+",
        default=["airline", "retail", "telecom"],
        help="Domains to analyze"
    )
    cl_requirements_parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Target memory buffer size"
    )
    cl_requirements_parser.set_defaults(func=lambda args: run_cl_data_requirements(args))

    # CL generate-splits command
    cl_splits_parser = subparsers.add_parser(
        "cl-generate-splits",
        help="Generate CL-specific data splits"
    )
    cl_splits_parser.add_argument(
        "--domains",
        nargs="+",
        default=["airline", "retail", "telecom"],
        help="Domains to generate splits for"
    )
    cl_splits_parser.add_argument(
        "--strategy",
        type=str,
        choices=["sequential", "interleaved", "difficulty"],
        default="sequential",
        help="Curriculum strategy"
    )
    cl_splits_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training set ratio"
    )
    cl_splits_parser.add_argument(
        "--num-phases",
        type=int,
        default=3,
        help="Number of learning phases"
    )
    cl_splits_parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/tau2/cl_splits",
        help="Output directory for split files"
    )
    cl_splits_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    cl_splits_parser.set_defaults(func=lambda args: run_cl_generate_splits(args))

    # CL generate-tasks command (LLM-based task generation)
    cl_generate_tasks_parser = subparsers.add_parser(
        "cl-generate-tasks",
        help="Generate synthetic training tasks using LLM"
    )
    cl_generate_tasks_parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["airline", "retail", "both"],
        help="Domain to generate tasks for"
    )
    cl_generate_tasks_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to generate (default: 307 for airline, 243 for retail)"
    )
    cl_generate_tasks_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for generation"
    )
    cl_generate_tasks_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: tasks_augmented.json in domain folder)"
    )
    cl_generate_tasks_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for concurrent API calls"
    )
    cl_generate_tasks_parser.set_defaults(func=lambda args: run_cl_generate_tasks(args))


def run_cl_generate_tasks(args):
    """Generate synthetic tasks using LLM."""
    from pathlib import Path
    from tau2.continual_learning.data_generation.llm_task_generator import (
        generate_airline_tasks,
        generate_retail_tasks,
    )

    output_path = Path(args.output) if args.output else None

    if args.domain == "airline" or args.domain == "both":
        num = args.num_tasks or 307
        print(f"Generating {num} airline tasks using {args.model}...")
        tasks = generate_airline_tasks(num, output_path, args.model)
        print(f"\n✓ Successfully generated {len(tasks)} airline tasks!")

    if args.domain == "retail" or args.domain == "both":
        num = args.num_tasks or 243
        print(f"Generating {num} retail tasks using {args.model}...")
        tasks = generate_retail_tasks(num, output_path, args.model)
        print(f"\n✓ Successfully generated {len(tasks)} retail tasks!")

    print("\nTask generation complete!")


def run_cl_validate_data(args):
    """Validate task data for CL experiments."""
    from pathlib import Path
    from tau2.continual_learning.data_generation.validator import validate_task_file

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file() and path.suffix == ".json":
            validate_task_file(path, domain=args.domain, verbose=True)
        elif path.is_dir():
            # Find all tasks.json in directory
            for tasks_file in path.rglob("tasks.json"):
                validate_task_file(tasks_file, domain=args.domain, verbose=True)
        else:
            print(f"Skipping invalid path: {path}")


def run_cl_data_requirements(args):
    """Analyze data requirements for CL experiments."""
    import json
    from pathlib import Path
    from tau2.continual_learning.data_generation.generator import print_data_requirements

    # Find task files
    base_path = Path("data/tau2/domains")
    domain_tasks = {}

    for domain in args.domains:
        tasks_file = base_path / domain / "tasks.json"
        if tasks_file.exists():
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            domain_tasks[domain] = tasks
        else:
            print(f"Warning: tasks.json not found for domain '{domain}'")

    if domain_tasks:
        print_data_requirements(domain_tasks, args.buffer_size)
    else:
        print("No task files found!")


def run_cl_generate_splits(args):
    """Generate CL-specific data splits."""
    import json
    from pathlib import Path
    from tau2.continual_learning.data_generation.generator import (
        CLSplitConfig,
        generate_cl_split,
        generate_multi_domain_cl_split,
    )

    base_path = Path("data/tau2/domains")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CLSplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=(1.0 - args.train_ratio) / 2,
        test_ratio=(1.0 - args.train_ratio) / 2,
        num_phases=args.num_phases,
        use_difficulty_ordering=(args.strategy == "difficulty"),
        seed=args.seed,
    )

    # Generate per-domain splits
    domain_files = {}
    for domain in args.domains:
        tasks_file = base_path / domain / "tasks.json"
        if tasks_file.exists():
            domain_files[domain] = tasks_file
            output_file = output_dir / f"{domain}_cl_split.json"
            generate_cl_split(
                tasks_file=tasks_file,
                output_file=output_file,
                strategy=args.strategy,
                config=config,
                domain=domain,
            )
        else:
            print(f"Warning: tasks.json not found for domain '{domain}'")

    # Generate combined multi-domain split
    if len(domain_files) > 1:
        combined_output = output_dir / "multi_domain_cl_split.json"
        generate_multi_domain_cl_split(
            domain_tasks_files=domain_files,
            output_file=combined_output,
            strategy=args.strategy,
            config=config,
        )

    print(f"\nSplits saved to: {output_dir}")


def print_cl_info():
    """Print information about the CL module."""
    print("""
================================================================
           Tau2-CL: Continual Learning for Tool Use
================================================================

  Continual Learning Framework for Parameter-Free API Agents

  FEATURES:
  - In-Context Learning with Experience Replay (ICL-ER)
  - Prompt Strategy Evolution (PSE)
  - Multiple curriculum strategies (Sequential, Interleaved, etc)
  - Comprehensive CL metrics (Forgetting, Transfer, etc)

  USAGE:
  1. Run experiment:
     tau2 cl-run --domains airline retail --curriculum sequential

  2. With config file:
     tau2 cl-run --config experiments/my_config.yaml

  3. Analyze results:
     tau2 cl-analyze experiments/results.json

  See docs/continuous_learning_framework_design.md for details

================================================================
""")


if __name__ == "__main__":
    main()
