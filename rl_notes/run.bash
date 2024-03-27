#!/bin/bash

TIMES=20

echo "Original another basic"
for i in `seq 1 $TIMES`; do
    python3 tests/original_another_basic.py 0
done

#echo "Last working another basic"
#for i in `seq 1 $TIMES`; do
#    python3 tests/last_working_another_basic.py 0
#done


echo "Another basic"
for i in `seq 1 $TIMES`; do
    python3 tests/another_basic.py 0
done
