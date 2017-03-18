# new_model_directory=$(date +%Y-%m-%d-%H:%M:%S)
# mkdir ../training_results/$new_model_directory
# printf "training network with: eta = 0.002 ------------------------------------"
# python test.py 0.002 $new_model_directory > ../training_results/$new_model_directory/training_printout.txt
# printf "\n\n\n\n\n"

new_model_directory=$(date +%Y-%m-%d-%H:%M:%S)
mkdir ../training_results/$new_model_directory
printf "training network with: eta = 0.002 ------------------------------------"
python network3_train.py 0.002 $new_model_directory > ../training_results/$new_model_directory/training_printout.txt
printf "\n\n\n\n\n"

new_model_directory=$(date +%Y-%m-%d-%H:%M:%S)
mkdir ../training_results/$new_model_directory
printf "training network with: eta = 0.003 ------------------------------------"
python network3_train.py 0.003 $new_model_directory > ../training_results/$new_model_directory/training_printout.txt
printf "\n\n\n\n\n"
