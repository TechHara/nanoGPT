python train.py config/train_shakespeare_char.py \
--device=cpu \
--compile=False \
--eval_iters=20 \
--log_interval=100 \
--block_size=256 \
--batch_size=12 \
--n_layer=4 \
--n_head=4 \
--n_embd=128 \
--max_iters=5000 \
--lr_decay_iters=2000 \
--dropout=0.0
