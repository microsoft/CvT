#!/bin/bash

train() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/train.py ${EXTRA_ARGS}
}


test() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/test.py ${EXTRA_ARGS}
}

############################ Main #############################
GPUS=`nvidia-smi -L | wc -l`
MASTER_PORT=9000
INSTALL_DEPS=false

while [[ $# -gt 0 ]]
do

key="$1"
case $key in
    -h|--help)
    echo "Usage: $0 [run_options]"
    echo "Options:"
    echo "  -g|--gpus <1> - number of gpus to be used"
    echo "  -t|--job-type <train> - job type (train|io|bit_finetune|test)"
    echo "  -p|--port <9000> - master port"
    echo "  -i|--install-deps - If install dependencies (default: False)"
    exit 1
    ;;
    -g|--gpus)
    GPUS=$2
    shift
    ;;
    -t|--job-type)
    JOB_TYPE=$2
    shift
    ;;
    -p|--port)
    MASTER_PORT=$2
    shift
    ;;
    -i|--install-deps)
    INSTALL_DEPS=true
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift
done

if $INSTALL_DEPS; then
    python -m pip install -r requirements.txt --user -q
fi

RANK=0
MASTER_ADDR=127.0.0.1
NODE_COUNT=1
echo "job type: ${JOB_TYPE}"
echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"

case $JOB_TYPE in
    train)
    train
    ;;
    test)
    test
    ;;
    *)
    echo "unknown job type"
    ;;
esac
