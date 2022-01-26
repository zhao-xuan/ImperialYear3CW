OMP_NUM_THREADS=1 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=2 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=3 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=4 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=8 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=12 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=16 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
sleep 10
OMP_NUM_THREADS=32 ./mttkrp -i ./tensors/nips.tns -o output.txt -d -1 -v nips-one-decimal.txt
