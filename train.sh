# "cartpole" "pendulum" 
declare -a envs=("cartpole" "pendulum" "dcmotor" "tape" "magnetic_pointer" "suspension" "biology" "cooling" "satellite" "quadcopter" "lane_keeping" "self_driving" "oscillator")

for env in "${envs[@]}"
do
    python3 DDPG.py --env=$env --train
done