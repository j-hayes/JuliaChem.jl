#!/bin/bash
#
source /etc/profile.d/modules.sh
source ~/.bashrc
#
export OMP_NUM_THREADS=111
#
cd /home/davpoolechem/shared/gms-dp/gamess
./rungms-dev S22_3/6-311++G_2d_2p/indole_benzene_2_t_shaped.inp 00 1 1
