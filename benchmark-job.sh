#! /usr/bin/bash

#SBATCH -N1
#SBATCH -n1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

set -ex
# The job must be executed under a TRT-LLM clone root folder

env && hostname && nvidia-smi
start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=benchmark.run.${SLURM_JOB_ID}.$(date '+%Y-%m-%d-%H:%M:%S')

clone_code() {
    unset GIT_LFS_SKIP_SMUDGE
    git fetch origin && git rebase origin/main
    git lfs pull
    git log --oneline origin/main | head -5
}

build() {
    bash ~/bin/dev-tekit-d.sh ~/bin/ctrt-llm.sh --clean
}

run_benchmark() {
    # bash ~/bin/dev-tekit-d.sh ./run_benchmark_bench.sh ${output_folder}
    bash ~/bin/dev-tekit-d.sh ./run_benchmark_serve.sh ${output_folder}.serve
}

report() {
    results=$1
    echo "Performance report ${SLURM_JOB_ID}"
    echo "==========================================="
    echo "Report path" $(realpath ${results})
    echo "START" $start_time "-" "END" ${end_time} $(hostname)
    grep -Hn "Per GPU Output Throughput" ${results}/*log || true
    grep -Hn "Output token throughput (tok/s):" ${results}.serve/*log || true
    echo "==========================================="
}

clone_code

mkdir -p ${output_folder}

echo "Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
git commit: $(git log --oneline origin/main | head -1)
output: ${output_folder} " | ~/bin/slack.sh

build
run_benchmark
end_time=$(date '+%Y-%m-%d-%H:%M:%S')

report ${output_folder} | ~/bin/slack.sh
