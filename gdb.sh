#!/bin/bash
# set -x

GDB=/usr/local/cuda/bin/cuda-gdb
command -v $GDB >/dev/null || { echo "cuda-gdb not found"; exit 1; }

if [ $# -ge 1 ]; then
    LEVEL=-$1
else
    LEVEL=-9
fi

if [ $# -ge 2 ]; then
    FILE=$2
else
    FILE=corpus/dickens
fi

rm -f $FILE.gz
$GDB --eval-command='b main' --eval-command='r' --args pigz/pigz -k $LEVEL $FILE
