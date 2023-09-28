#!/bin/bash

set -e

eval "$(conda shell.bash hook)"

export BENCH_DIR=numba-mlir-bench
export GH_BRANCH=gh-pages
export GH_PAGE=gh-pages

rm -rf ${BENCH_DIR}
mkdir ${BENCH_DIR}
pushd ${BENCH_DIR}
git clone -b ${GH_BRANCH} https://${GH_PAGES_USER}:${GH_PAGES_TOKEN}@github.com/${GH_PAGES_USER}/numba-mlir.git ${GH_PAGE}
git clone https://github.com/numba/numba-mlir.git

export ENV_NAME="test-asv"
conda remove -n ${ENV_NAME} --all -y || true
conda env create -n ${ENV_NAME} -f ./numba-mlir/scripts/bench-env-linux.yml
conda activate ${ENV_NAME}
conda list

mkdir ./${GH_PAGE}/asv/html || true
mkdir ./${GH_PAGE}/asv/results || true

export NUMBA_MLIR_COMMIT=`python -c "import numba_mlir; print(numba_mlir._version.get_versions()['full-revisionid'])"`
export PAGE_DIR=`cd ./${GH_PAGE}/asv/html; pwd`
export NUMBA_MLIR_BENCH_RUNNER_RESULTS_DIR=`cd ./${GH_PAGE}/asv/results; pwd`
export NUMBA_MLIR_BENCH_RUNNER_MACHINE=$1

echo "Machine: ${NUMBA_MLIR_BENCH_RUNNER_MACHINE}"
echo "Results dir: ${NUMBA_MLIR_BENCH_RUNNER_RESULTS_DIR}"
echo "Page dir: ${PAGE_DIR}"

pushd numba-mlir
git checkout ${NUMBA_MLIR_COMMIT}
cd benchmarks

echo "Setup machine"
python runner.py machine

echo "Run benchmarks"
python runner.py bench $2

echo "Publish results"
python runner.py publish ${PAGE_DIR}
popd

cd ./${GH_PAGE}/
git config user.email "(none)"
git config user.name "Benchmark automation"
git add *
git commit -m "run benchmark"
git pull --rebase origin ${GH_BRANCH}
git push origin ${GH_BRANCH}

conda deactivate
conda remove -n ${ENV_NAME} --all -y
popd
rm -rf ${BENCH_DIR}
