---
status: accepted
date: 2026-11-26
---

# Define named experiments in Python

Use one Python module per experiment family. Keep a typed schema for each
workload with its model, data, training, evaluation and resource fields
discoverable together. Group reusable fields in dataclasses. Define a family
base as a frozen, keyword-only dataclass with the family's defaults. List
instances in `experiment_configs`, giving each a stable name and spelling out
only the fields that vary. Use dataclass inheritance to share defaults. Changes
to one instance must leave the others independent. Keep numeric settings in
code. The registry checks name uniqueness within each family.

## Worker interface

The launcher selects a family, run and phase, resolves the dataclass defaults
and builds an argv list from the fields consumed by that worker. The dry run
shows the effective arguments, resources and worker command. The worker parses
those same typed field definitions with `HfArgumentParser` and
`parse_args_into_dataclasses(look_for_args_file=False)`. Use distinct CLI names
for fields consumed together. Keep job allocation fields in the launcher.
Give evaluation exactly one GPU and keep probe and training allocations separate.

Migrate the pilot first, then other families and the direct paper entry points
for training, evaluation, generation and data preparation. Replace Hydra model
instantiation with explicit construction from typed model arguments. Port
paper recipes, output directory selection and logging defaults to Python.
Remove the Hydra and YAML experiment configuration path once the replacement
commands reproduce the existing effective settings. All worker configuration
travels as CLI arguments, including when launched directly.

## Idempotent submission

Each independently submitted run or phase declares its final absolute
`output_dir` in its Python config. This path is its submission identity. To
request another result, choose a new path. Existing output directories are
skipped on existence alone, including after changes to code or arguments.

Keep a shared `in_progress/` tree on the same artifact filesystem. Mirror the
final path relative to the artifacts root so equal basenames under different
parents have independent reservations. For each submission:

1. Skip if the final directory exists or its corresponding reservation exists.
2. Atomically create the reservation directory before calling MLSpace. If
   another launcher wins the mkdir, skip. Recheck the final path after acquiring
   the reservation to cover a concurrent completion. Dry runs only inspect.
3. Record the full MLSpace job name in the reservation once available. Write
   intermediate results under the reservation. On success, atomically rename
   its result subdirectory to the final path, then release the reservation.
4. Release the reservation after a confirmed submission rejection, failure or
   cancellation, preserving any partial results separately. After an ambiguous
   submission or a launcher crash, keep the reservation until reconciliation.
   Use the recorded job name or the scaffold's active-job lookup by normalized
   description to identify the job. An unresolved reservation continues to
   block submission.

DFlash publishes its checkpoint directory upon training completion. Both final and
milestone evaluations require the published training result and the requested weights.

## Migration checks

Compare old and new effective settings and worker inputs for each migrated
phase. Round-trip launcher argv through the actual worker parser, covering
optional values, booleans and lists. Map existing results and active jobs to
their configured destinations before enabling the new submission path. Exercise
existing outputs, concurrent reservations, successful publication, confirmed
failures and ambiguous submission outcomes with a fake MLSpace client.
