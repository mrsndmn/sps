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

Before submitting, resolve the effective configuration and inputs for the phase.
Associate the family, run, phase and design signature with its artifacts and
submission receipt. Include inputs that can change the result, such as the
dataset revision, training checkpoint, evaluation cohort and generation
settings. Changing an unrelated run must preserve this run's identity.

- Return the existing full MLSpace job name for a matching queued or running job.
- Skip submission when the matching completion marker and required artifacts
  establish that the phase succeeded.
- Surface an incompatible signature at an existing artifact destination. A
  changed design needs a distinct destination before it can run.
- Allow a retry after confirmed failure or cancellation. Reconcile a pending
  submission intent with MLSpace after a crash. If acceptance is uncertain,
  report the ambiguity and hold submission until it is resolved.
- Serialize competing submissions using the existing lock and durable intent
  and receipt protocol. Apply the same guarantees to specialized launchers.

## Migration checks

Compare old and new effective settings and worker inputs for each migrated
phase. Round-trip launcher argv through the actual worker parser, covering
optional values, booleans and lists. Verify reuse of compatible historical
receipts, checkpoints and completion markers. Exercise sequential and concurrent
duplicate submissions, failures, ambiguous submission outcomes and signature
mismatches with a fake MLSpace client before relying on each launcher route.
