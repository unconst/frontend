pm2 start trainer.py -f --interpreter python3 --name trainer -- --device cuda:0 --grads-per-step 6 --learning-rate 0.000005
pm2 start miner.py -f --interpreter python3 --name miner1 -- --name miner1 --device cuda:1 --use-wandb
pm2 start miner.py -f --interpreter python3 --name miner2 -- --name miner2 --device cuda:2 --use-wandb
pm2 start miner.py -f --interpreter python3 --name miner3 -- --name miner3 --device cuda:3 --use-wandb
pm2 start miner.py -f --interpreter python3 --name miner4 -- --name miner4 --device cuda:4 --use-wandb
pm2 start miner.py -f --interpreter python3 --name miner5 -- --name miner5 --device cuda:5 --use-wandb
pm2 start miner.py -f --interpreter python3 --name miner6 -- --name miner6 --device cuda:6 --use-wandb
