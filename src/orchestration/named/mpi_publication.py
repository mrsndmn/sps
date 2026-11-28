"""Run one worker per MPI rank and publish only after every worker succeeds."""
import os
import re
import subprocess
import time


def mpi_environment(env):
    nodes = int(env["SPS_MPI_NODES"])
    per_node = int(env["SPS_MPI_GPUS_PER_NODE"])
    world = int(env["OMPI_COMM_WORLD_SIZE"])
    rank = int(env["OMPI_COMM_WORLD_RANK"])
    local = int(env["OMPI_COMM_WORLD_LOCAL_RANK"])
    if world != nodes * per_node or not (0 <= rank < world and 0 <= local < per_node):
        raise RuntimeError(f"Invalid MPI ranks {rank}/{world}, local {local} for {nodes}x{per_node}")
    host = re.fullmatch(r"([\w-]+)-\w+-\d+", env["PMIX_HOSTNAME"])
    if host is None:
        raise RuntimeError(f"Unexpected MLSpace hostname {env['PMIX_HOSTNAME']}")
    job = host[1]
    return dict(env, WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(local),
                MASTER_ADDR=f"{job}-mpimaster-0.{job}", MASTER_PORT="12345")


def run_mpi(directory, command, env):
    env = mpi_environment(env)
    if not directory.reservation.is_dir():
        raise RuntimeError(f"Missing submission reservation {directory.reservation}")
    directory.work.mkdir(exist_ok=True)
    # The reservation belongs to this submission. Retried jobs acquire a new one.
    sync = directory.reservation / "rank-exits"
    sync.mkdir(exist_ok=True)
    rank = int(env["RANK"])
    env.update(SPS_JOB_OUTPUT=str(directory.output), SPS_JOB_WORK=str(directory.work))
    subprocess.run(["/bin/bash", "-c", command], env=env, check=True)
    (sync / str(rank)).touch()
    if rank == 0:
        expected = int(env["WORLD_SIZE"])
        deadline = time.monotonic() + 300
        while len(list(sync.iterdir())) != expected:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for MPI workers to finish")
            time.sleep(1)
        directory.finish()
