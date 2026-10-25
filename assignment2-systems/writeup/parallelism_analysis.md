#  Analyzing Parallelism Strategies


## 8.1
```
(S / W) * (N-1)
```

## 8.2
```
(a) backward: 12 * (B / N_DP) * D * D_FF

(b) 12 * D * D_FF * (N_DP−1) / (N_DP * W)

(c) N_DP <= W*B/C + 1; =>   ~ N_DP <= B * (W/C)  => B/N_DP >= C/W
```


## 8.3
```
(a) backward: 12 * (B / N_FSDP) * D * D_FF
    forward:  6 * (B / N_FSDP) * D * D_FF

(b) backward: 12 * D * D_FF * (N_FSDP-1) / (N_FSDP * W)

all_gather + reduce_scatter
all_gather : (N-1) * ((W1+W2+W3)/N ) / W = (N-1)*3*D*D_FF / (N*W)  * 2 (fp16)
educe_scatter: (N-1) * (dW1 + dW2 + dW3)/N / W = (N-1)*3*D*D_FF/ (N*W)  * 2 (fp15)

(c)  N_FSDP <= W*B/C + 1;  SAME AS DP!  => B/N_DP >= C/W

forward
6 * B * D * D_FF / (N_FSDP * C) >= 6 * D * D_FF * (N_FSDP-1) / (N_FSDP * W)

backward
12 * B * D * D_FF / (N_FSDP * C) >= 12 * D * D_FF * (N_FSDP-1) / (N_FSDP * W)

both are same,
=> B / C >=  (N_FSDP-1) / W
=> N_FSDP <= B * (W/C) + 1
```


## 8.4
```
Notation:
- 'W.T' means transpose of W,
- 'W(i)' is i-s hard of W. Can be column parallel or row parallel, depends on the shape
- '@' means matmul
- '*' means element-wise multiply

(a)
dz(i) = dy @ W3.T(i)
dx2(i) = dz(i) * f(x1(i))
dx1(i) = dz(i) * f'(x1(1)) * x2(i)

dx(i) = dx1(i) @ W1.T(i) + dx2(i) @ W2.T(i)
dx = all_reduce(dx(i))

# assume activation z.T  is stored during forward
dW3(i) = z.T(i) @ dy
dW2(i) = x.T @ dx2(i)
dW1(i) = x.T @ dx1(i)

(b)
Forward FLOPs:  6*B*D*D_FF/N_TP
Backward FLOPs: 12*B*D*D_FF/N_TP

(c)
fp16 = 2 Bytes
Forward: 1 all_reduce = 4*(N_TP-1)*B*D/(N_TP*W)
Backward: 1 all_reduce = 4*(N_TP-1)*B*D/(N_TP*W)

(d)
Forward: N_TP <= (3*W*D_FF)/ (2*C) + 1
Backward: N_TP <= (3*W*D_FF)/ C + 1
=> N_TP <= (3*W*D_FF)/ (2*C) + 1
=> N_TP <= 1.5 * D_FF * (W/C) + 1
```

## 8.5
```
(a) FLOPs
6 * B * D * D_FF / (N_FSDP * N_TP)

(b)
fp16 = 2 Bytes

FSDP term:  3 all_gather
3*(N_FSDP-1)*D*D_FF*2 / (N_FSDP * N_TP * W)

TP term: 1 all_reduce
y(i,j) shpae (B/N_FSDP, D)
2*(N_TP-1)*B*D*2 / (N_FSDP * N_TP * W)

max from two
max(6*(N_FSDP-1)*D_FF, 4*(N_TP-1)*B) * D / (N_FSDP * N_TP *W)

(c)
max(6*(N_FSDP-1)*D_FF, 4*(N_TP-1)*B) * D / (N_FSDP * N_TP *W) <= 6 * B * D * D_FF / (N_FSDP * N_TP * C)
=>
max(6*(N_FSDP-1)*D_FF, 4*(N_TP-1)*B)/ W <= 6 * B * D_FF / C
=>
max(3*(N_FSDP-1)*D_FF, 2*(N_TP-1)*B) <= 3*B*D_FF * (W/C)

=> N_FSDP-1 <= B * (W/C),
   N_TP-1 <= 1.5*D_FF * (W/C)

=> (N_FSDP-1)(N_TP-1) <= 1.5*B_D_FF * (W/C) * (W/C)

for large B*(W/C) >> 1, D_FF*(W/C) >> 1
=> N_FSDP * N_TP <= 1.5 * B * D_FF * (W/C) * (W/C)
=> N <= 1.5*B*D_FF*(W/C)^2

(d)
sum(6*(N_FSDP-1)*D_FF, 4*(N_TP-1)*B) * D / (N_FSDP * N_TP *W) <=  6 * B * D * D_FF / (N_FSDP * N_TP * C)
=>
3*(N_FSDP-1)*D_FF + 2*(N_TP-1)*B  <=  3*B*D_FF*(W/C)

for large-N,
=>
3*D_FF* N_FSDP + 2*B* N_TP <= 3*B*D_FF*(W/C)

Find max(x*y) for:
  a*x + b*y <= c

=> x = c/2a, y = c/2b, max(x*y) = c^2/4ab
use c = 3*B*D_FF*(W/C), a = 3*D_FF, b = 2*B
=> max(N_TP*N_FSDP) = 9*B^2*D_FF^2*(W/C)*(W/C) / (4*3*D_FF*2*B)
=> N_TP*N_FSDP <=  3*B*D_FF*(W/C)^2 / 8
=> N <= 0.375*B*D_FF*(W/C)^2

```
