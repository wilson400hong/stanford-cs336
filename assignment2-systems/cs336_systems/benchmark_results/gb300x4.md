# small
```bash
uv run python -m cs336_systems.benchmark --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12
```
[Forward] mean=0.023898, std=0.000607
[Backard] mean=0.031660, std=0.000725
[Optimizer] mean=0.025553, std=0.000763

```bash
uv run python -m cs336_systems.benchmark --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 -mp bfloat16
```
[Forward] mean=0.023785, std=0.000772
[Backard] mean=0.032979, std=0.000305
[Optimizer] mean=0.022764, std=0.000858

# medium
```bash
uv run python -m cs336_systems.benchmark --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16
```
[Forward] mean=0.053170, std=0.013457
[Backard] mean=0.088810, std=0.000265
[Optimizer] mean=0.044542, std=0.005225


# large
```bash
uv run python -m cs336_systems.benchmark --d_model 1280 --d_ff 5120 --num_layers 36 --num_heads 20
```
[Forward] mean=0.104013, std=0.004039
[Backard] mean=0.198484, std=0.000903
[Optimizer] mean=0.073690, std=0.004475

# xl
```bash
uv run python -m cs336_systems.benchmark --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32
```
[Forward] mean=0.278723, std=0.000403
[Backard] mean=0.535158, std=0.000794
[Optimizer] mean=0.100517, std=0.001166

# 10B
```bash
uv run python -m cs336_systems.benchmark --d_model 4608 --d_ff 12288 --num_layers 50 --num_heads 36
```
[Forward] mean=0.868847, std=0.000559
[Backard] mean=1.777293, std=0.001997
[Optimizer] mean=0.341641, std=0.000234

```bash
uv run python -m cs336_systems.benchmark --d_model 4608 --d_ff 12288 --num_layers 50 --num_heads 36 -mp bfloat16
```
[Forward] mean=0.117710, std=0.002268
[Backard] mean=0.268555, std=0.003528
[Optimizer] mean=0.340577, std=0.000196


### NSys
```bash
uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0   -- python -m cs336_systems.benchmark --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12


# nvtx  + memory profile
uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop   --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx   -- python -m cs336_systems.benchmark --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12   --mem_prof_file small_vanilla
```

## activation checkpoint
### small
```bash
uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop   --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx   -- python -m cs336_systems.benchmark  --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12   --mem_prof_file small_vanilla

uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop   --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx   -- python -m cs336_systems.benchmark --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12   --mem_prof_file small_gc3  --gradient_checkpointing --layer_chunk_size 3
```

### xl - 2048
```bash
uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop   --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx   -- python -m cs336_systems.benchmark  --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32  --context_length 2048 --mem_prof_file xl_vanilla


uv run nsys profile -o ~/gdrive/nsight/myrun --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop   --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx   -- python -m cs336_systems.benchmark  --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32  --context_length 2048 --mem_prof_file xl_gc1 --gradient_checkpointing --layer_chunk_size 1

```
