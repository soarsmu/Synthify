declare -a envs=("cartpole" "pendulum" "quadcopter" "self_driving" "lane_keeping" "car_platoon_4" "car_platoon_8" "oscillator")

# declare -a envs=("self_driving")

mkdir results

for num in {1..10}
do
  for env in "${envs[@]}"
  do
    python3 falsify_time_budget.py --env=$env \
    2>&1 | tee results/s2f_"$env"_"$num"_time.log &
  done
done