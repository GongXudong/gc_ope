#!/bin/bash
source ~/ai4robot/gc_ope/.venv/bin/activate
file=/home/maxine/ai4robot/gc_ope/tests/algorithm/ope/test_ope.py

## test all algo, env
# python $file env=my_reach env_id=MyReachSparse-v0 algo=sac
# python $file env=my_reach env_id=MyReachSparse-v0 algo=her
# python $file env=my_slide env_id=MySlideSparse-v0 algo=sac
# python $file env=my_slide env_id=MySlideSparse-v0 algo=her
# python $file env=my_push env_id=MyPushSparse-v0 algo=sac #似乎比较费时
# python $file env=my_push env_id=MyPushSparse-v0 algo=her
python $file env=flycraft env_id=FlyCraft-v0 algo=sac
# python $file env=flycraft env_id=FlyCraft-v0 algo=her


# python $file env=my_reach env_id=MyReachSparse-v0 algo=omega
# python $file env=my_slide env_id=MySlideSparse-v0 algo=omega #done
# python $file env=my_push env_id=MyPushSparse-v0 algo=omega #done
# python $file env=flycraft env_id=FlyCraft-v0 algo=omega #done