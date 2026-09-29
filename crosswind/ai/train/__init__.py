"""Training scripts for the computer player (need requirements-train.txt)."""
import os

# The scripts run one process per core; numpy's tiny matrix products must not each start a thread pool
# on top of that (with 20 threads per process, training ran ~80x slower).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
