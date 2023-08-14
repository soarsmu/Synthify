declare -a envs=("cartpole" "pendulum" "quadcopter" "self_driving" "lane_keeping" "car_platoon_4" "car_platoon_8" "oscillator")

mkdir results

# for num in {1..10}
# do
for env in "${envs[@]}"
do
  python3 falsify.py --env=$env \
  2>&1 | tee results/s2f_discussion_"$env"_"$num".log &
done
# done


# declare -a envs=("quadcopter")

# mkdir results

# for num in {1..10}
# do
#   for env in "${envs[@]}"
#   do
#     python3 falsify.py --env=$env \
#     2>&1 | tee results/s2f_"$env"_"$num".log &
#   done
# done