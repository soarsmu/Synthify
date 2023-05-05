declare -a envs=("cartpole" "pendulum" "quadcopter" "self_driving")

mkdir results

for num in {1..20}
do
  for env in "${envs[@]}"
  do
    python3 falsify_time_budget.py --env=$env \
    2>&1 | tee results/s2f_"$env"_"$num"_time.log
  done
done