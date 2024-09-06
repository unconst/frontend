

pm2 start train.py -f --interpreter python3 --name seq1 -- --device cuda:0 --method seqcompress --learning-rate 0.001 --batch-size 12 --use-wandb --run-name seq1 
pm2 start train.py -f --interpreter python3 --name seq2 -- --device cuda:1 --method seqcompress --learning-rate 0.005 --batch-size 12 --use-wandb --run-name seq2 
pm2 start train.py -f --interpreter python3 --name seq3 -- --device cuda:2 --method seqcompress --learning-rate 0.0001 --batch-size 12 --use-wandb --run-name seq3 
pm2 start train.py -f --interpreter python3 --name seq4 -- --device cuda:3 --method seqcompress --learning-rate 0.0005 --batch-size 12 --use-wandb --run-name seq4 
pm2 start train.py -f --interpreter python3 --name seq5 -- --device cuda:4 --method seqcompress --learning-rate 0.00001 --batch-size 12 --use-wandb --run-name seq5 
pm2 start train.py -f --interpreter python3 --name seq6 -- --device cuda:5 --method seqcompress --learning-rate 0.000005 --batch-size 12 --use-wandb --run-name seq6 
pm2 start train.py -f --interpreter python3 --name seq7 -- --device cuda:6 --method seqcompress --learning-rate 0.000001 --batch-size 12 --use-wandb --run-name seq7 
pm2 start train.py -f --interpreter python3 --name seq8 -- --device cuda:7 --method seqcompress --learning-rate 0.0000005 --batch-size 12 --use-wandb --run-name seq8
