#!/bin/bash

screen -dmS hokuyo /run-hokuyo.sh
screen -dmS f1tenth /run.sh
sleep 5
screen -dmS mouse /run-mouse.sh
screen -dmS sagol /run-sagol.sh
exec bash
