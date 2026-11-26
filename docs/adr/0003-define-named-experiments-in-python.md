---
status: accepted
date: 2026-11-26
---

# Define named experiments in Python

Define each named experiment and its effective settings in Python. A family
can share defaults across its runs. Collect the run definitions in a registry
that training and evaluation launchers can read. Workers can consume a
generated configuration file. Change the Python definition to change a run.

Use one Python module per experiment family. Its function returns named runs
and makes the differences between variants explicit. Shared Python recipes
define reusable settings. The registry collects the families and checks for
duplicate run names. A dry run shows the effective settings and worker command
for the selected name.

The pilot previously combined Hydra recipes with Python command-line overrides.
This required readers to trace both sources to determine the training budget.
Using one authoring language lets readers inspect the settings before submission.
Each migrated family must preserve the settings that its workers receive.

On 2026-11-26 the user selected Python as the only place to author experiment
definitions and selected a staged migration for all families.
