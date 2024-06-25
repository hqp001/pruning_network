#!/bin/bash

./run_single_instance.sh "v1" "./" "mnist_fc" "mnist-net_256x2.onnx" "./mnist_fc/vnnlib/prop_1_0.03.vnnlib" 100 "results.csv" "couterexample"
