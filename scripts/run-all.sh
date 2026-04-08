#!/bin/bash
screen -wipe
screen -dmS f1tenth /run.sh
sleep 5
screen -dmS sagol /run-sagol.sh
screen -dmS hotrc /run-hotrc.sh
journalctl -f

