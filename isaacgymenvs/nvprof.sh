#!/bin/bash

CMD="python train.py task=FrankaCabinet headless=True max_iterations=25"

nvprof -f -o net.sql --profile-from-start off $CMD
python -m pyprof.parse net.sql > net.dict
python -m pyprof.prof --csv net.dict -c idx,dir,op,kernel,params,sil,tc,flops,bytes > net.csv
python count_csv.py
