#!/bin/bash

file=/home/maxine/ai4robot/gc_ope/tests/algorithm/ope/test_ope_eval.py

uv run --offline $file env=my_reach env_id=MyReachSparse-v0 algo=omega