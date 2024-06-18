#!/bin/bash

nohup python3 fednoro_standalone.py --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0 --aggregator fednoro --noisy_selection True > log_fednoro_cs.out &
wait
nohup python3 fednoro_standalone.py --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0 --aggregator fednoro --noisy_selection False > log_fednoro.out &
wait
nohup python3 fednoro_standalone.py --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0 --aggregator fedavg --noisy_selection True > log_fedavg_cs.out &
wait
nohup python3 fednoro_standalone.py --warm_up_round 10 --total_client 20 --com_round 100 --level_n_system 0 --aggregator fedavg --noisy_selection False > log_fedavg.out &
wait
