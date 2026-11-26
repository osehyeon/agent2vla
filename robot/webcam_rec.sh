#!/bin/bash
# usage: webcam_rec.sh MINUTES   (0 = stop now)
echo $(( $(date +%s) + ${1:-5} * 60 )) > "${AGENT2VLA_WORK:-$HOME/lerobot-work}/webcam/REC"
