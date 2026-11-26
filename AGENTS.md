# Project guidance

Keep experiment definitions and numeric configuration in code. Experiment Markdown
contains only Hypothesis, Setup (training function), Results, Conclusions and Changelog.
Stage files explicitly and commit implementation changes with conventional prefixes.
Do not modify the shared mls library. Use the existing project Python environment.
For large benchmark evaluations use vllm (see existing scripts).
Report complete MLSpace UUID job names. Long jobs belong in MLSpace, with durable
Arkhip job_wait observation rather than polling shells.

Describe every launchable phase of each new experiment family, including data
preparation and training, with dataclass instances. Define a family dataclass
with a `@classmethod def experiment_configs(cls)` that returns a list containing
every run. Put phase-specific settings and dependencies on those instances.
The named CLI, MLSpace launcher, and workers must derive their selections from
that method. Use `PaperSPS.experiment_configs()` in
`src/experiments/paper_reproduction.py` as the example.
