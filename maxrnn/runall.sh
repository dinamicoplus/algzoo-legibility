#!/bin/bash
cd "$(dirname "$0")"
( python3 train.py relu 16 20000 0 > log_relu16_0.txt 2>&1; python3 train.py max 16 20000 1 > log_max16_1.txt 2>&1; python3 train.py relu 20 20000 0 > log_relu20_0.txt 2>&1 ) &
( python3 train.py max 16 20000 0 > log_max16_0.txt 2>&1; python3 train.py max 20 20000 0 > log_max20_0.txt 2>&1 ) &
wait
echo ALLDONE > done.txt
