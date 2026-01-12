#!/bin/bash

file=/home/maxine/ai4robot/gc_ope/tests/algorithm/ope/test_ope.py

## test all algo, env
uv run --offline $file env=my_reach env_id=MyReachSparse-v0 algo=sac
# uv run --offline $file env=my_reach env_id=MyReachSparse-v0 algo=her
# uv run --offline $file env=my_slide env_id=MySlideSparse-v0 algo=sac
# uv run --offline $file env=my_slide env_id=MySlideSparse-v0 algo=her
# uv run --offline $file env=my_push env_id=MyPushSparse-v0 algo=sac #似乎比较费时
# uv run --offline $file env=my_push env_id=MyPushSparse-v0 algo=her