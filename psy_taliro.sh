declare -a envs=("cartpole" "pendulum" "quadcopter" "self_driving" "lane_keeping" "car_platoon_4" "car_platoon_8" "oscillator")

mkdir results
for num in {1..10}
do
  for env in "${envs[@]}"
  do
    python3 psy_taliro.py --env=$env \
    2>&1 | tee results/psy_taliro_"$env"_"$num".log &
  done
done
