pm2 start topkf.py -f --interpreter python3 --name t1 -- --run-name t1 --device cuda:0 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name t2 -- --run-name t2 --device cuda:1 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name t3 -- --run-name t3 --device cuda:2 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name t4 -- --run-name t4 --device cuda:3 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name t5 -- --run-name t5 --device cuda:4 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name t6 -- --run-name t6 --device cuda:5 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk
pm2 start topkf.py -f --interpreter python3 --name baseline -- --run-name baseline --device cuda:6 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --method baseline


pm2 start topkf.py -f --interpreter python3 --name t1 -- --run-name t1 --device cuda:0 --model-name llama --learning-rate 0.0001 --batch-size 6 --use-wandb --compression 1000 --method topk --scoring hessian
