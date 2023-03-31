declare -a envs=("cartpole" "pendulum" "quadcopter" "self_driving")

mkdir results
for num in {1..20}
do
  for env in "${envs[@]}"
  do
    python3 psy_taliro.py --env=$env \
    2>&1 | tee results/baseline_"$env"_"$num".log
  done
done