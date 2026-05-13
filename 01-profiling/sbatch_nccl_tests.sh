#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "step1_nccl_tests"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=176
#SBATCH --gres=gpu:4
#SBATCH --threads-per-core=1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=00:20:00
#SBATCH --output=/home/fu92078/thesis/results/raw_logs/step1_nccl_tests/step1_nccl_tests_%j.out
#SBATCH --error=/home/fu92078/thesis/results/raw_logs/step1_nccl_tests/step1_nccl_tests_%j.err

# Run nccl-tests on one MUSICA node

set -euo pipefail

THESIS_ROOT=/home/fu92078/thesis
BENCH_DIR=${THESIS_ROOT}/benchmarks/nccl_tests
RAW_LOG_DIR=${THESIS_ROOT}/results/raw_logs/step1_nccl_tests

CONTAINER_TAG=25.04-py3
CONTAINER_SIF=${HOME}/containers/pytorch_${CONTAINER_TAG}.sif
NCCL_TESTS_SRC=${BENCH_DIR}/nccl-tests-src
NCCL_TESTS_BUILD=${NCCL_TESTS_SRC}/build

JOB_DIR=${RAW_LOG_DIR}/job_${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

unset LD_PRELOAD || true

if [[ ! -f ${CONTAINER_SIF} ]]; then
    echo "ERROR: container ${CONTAINER_SIF} not found." >&2
    echo "       run ${BENCH_DIR}/setup_nccl_tests_container.sh on the login node first." >&2
    exit 1
fi
if [[ ! -x ${NCCL_TESTS_BUILD}/all_reduce_perf ]]; then
    echo "ERROR: nccl-tests binaries not built in ${NCCL_TESTS_BUILD}." >&2
    echo "       run ${BENCH_DIR}/setup_nccl_tests_container.sh on the login node first." >&2
    exit 1
fi

echo "===== Job ${SLURM_JOB_ID} on $(hostname) at $(date -Is) ====="
echo "container:        ${CONTAINER_SIF}"
echo "nccl-tests build: ${NCCL_TESTS_BUILD}"
echo "nccl-tests commit:$(git -C ${NCCL_TESTS_SRC} rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "job dir:          ${JOB_DIR}"

echo "----- host nvidia-smi -----"
nvidia-smi || true
echo "----- host nvidia-smi topo -m -----"
nvidia-smi topo -m || true
echo "----- container NCCL version banner -----"
apptainer exec --nv ${CONTAINER_SIF} \
    bash -c 'python - <<PY
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version())
PY' || true

NCCL_TESTS_ARGS=(
    -b 1K
    -e 8G
    -f 2
    -g 4
    -n 50
    -w 10
    -c 1
)

COLLECTIVES=(
    all_reduce_perf
    all_gather_perf
    reduce_scatter_perf
    broadcast_perf
    reduce_perf
    alltoall_perf
    sendrecv_perf
)

echo "----- nccl-tests args: ${NCCL_TESTS_ARGS[*]} -----"

for COLL in "${COLLECTIVES[@]}"; do
    LOG=${JOB_DIR}/${COLL}.log
    BIN=${NCCL_TESTS_BUILD}/${COLL}
    if [[ ! -x ${BIN} ]]; then
        echo "[skip] ${COLL}: binary ${BIN} not found" | tee -a ${LOG}
        continue
    fi
    echo ""
    echo "===== ${COLL} @ $(date -Is) ====="
    {
        echo "# host=$(hostname) job=${SLURM_JOB_ID} date=$(date -Is)"
        echo "# binary=${BIN}"
        echo "# args=${NCCL_TESTS_ARGS[*]}"
        echo "# container=${CONTAINER_SIF}"
        echo "# nccl-tests commit=$(git -C ${NCCL_TESTS_SRC} rev-parse --short HEAD 2>/dev/null || echo unknown)"
    } > ${LOG}
    apptainer exec --nv ${CONTAINER_SIF} \
        ${BIN} "${NCCL_TESTS_ARGS[@]}" \
        2>&1 | tee -a ${LOG}
done

echo ""
echo "===== Job ${SLURM_JOB_ID} finished at $(date -Is) ====="
echo "logs: ${JOB_DIR}/"
ls -la ${JOB_DIR}/
