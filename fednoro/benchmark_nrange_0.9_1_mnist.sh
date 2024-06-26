#!/bin/bash

# Execute the first Python program and wait for it to complete
nohup python3 fednoro_standalone_mnist.py --dataname mnist --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0.4 --level_n_lowerb 0.9 --level_n_upperb 1 --aggregator fednoro --noisy_selection True > log_fednoro_cs.out 2>&1 &
echo "Started fednoro_standalone_mnist.py with --aggregator fednoro --noisy_selection True in the background with output in log_fednoro_cs.out"
wait $!
echo "fednoro_standalone.py with --aggregator fednoro --noisy_selection True has finished"

# Execute the second Python program and wait for it to complete
nohup python3 fednoro_standalone_mnist.py --dataname mnist --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0.4 --level_n_lowerb 0.9 --level_n_upperb 1 --aggregator fednoro > log_fednoro.out 2>&1 &
echo "Started fednoro_standalone_mnist.py with --aggregator fednoro --noisy_selection False in the background with output in log_fednoro.out"
wait $!
echo "fednoro_standalone.py with --aggregator fednoro --noisy_selection False has finished"

# Execute the third Python program and wait for it to complete
nohup python3 fednoro_standalone_mnist.py --dataname mnist --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0.4 --level_n_lowerb 0.9 --level_n_upperb 1 --aggregator fedavg --noisy_selection True > log_fedavg_cs.out 2>&1 &
echo "Started fednoro_standalone_mnist.py with --aggregator fedavg --noisy_selection True in the background with output in log_fedavg_cs.out"
wait $!
echo "fednoro_standalone.py with --aggregator fedavg --noisy_selection True has finished"

# Execute the fourth Python program and wait for it to complete
nohup python3 fednoro_standalone_mnist.py --dataname mnist --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0.4 --level_n_lowerb 0.9 --level_n_upperb 1 --aggregator fedavg > log_fedavg.out 2>&1 &
echo "Started fednoro_standalone_mnist.py with --aggregator fedavg --noisy_selection False in the background with output in log_fedavg.out"
wait $!
echo "fednoro_standalone.py with --aggregator fedavg --noisy_selection False has finished"

echo "All programs have been executed."
