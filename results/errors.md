fp32 GPT (4 layers, d = 128) after 300 steps (loss 2.39). SQNR in dB of each GEMM's output against fp32, median over the 16 projections (worst in brackets). Higher is better; 30 dB ≈ 3% relative error.

| recipe | forward x·Wᵀ | data grad dy·W | weight grad dyᵀ·x |
|---|---|---|---|
| bf16 | 63.6 (61.5) | 61.3 (59.1) | 59.5 (56.5) |
| fp8_tensor | 39.6 (37.1) | 33.4 (31.2) | 31.2 (28.0) |
| fp8_block | 39.7 (37.5) | 37.6 (35.3) | 35.9 (33.0) |
| mxfp8 | 37.1 (35.9) | 35.5 (33.9) | 31.9 (29.5) |
| mxfp4 | 24.0 (22.9) | 22.6 (21.6) | 17.8 (14.4) |
| mxfp4 + RHT | 24.0 (22.9) | 22.6 (21.6) | 20.8 (19.0) |
| nvfp4 | 28.3 (26.1) | 26.1 (23.9) | 22.4 (18.8) |
| nvfp4 + RHT | 28.3 (26.1) | 26.1 (23.9) | 23.8 (21.0) |

amax / rms of each operand (a Gaussian of this size gives about 4–5):

| projection | x | W | dy |
|---|---|---|---|
| blocks.0.qkv | 4.5 | 5.5 | 134.8 |
| blocks.0.proj | 3.5 | 4.7 | 13.7 |
| blocks.0.fc1 | 3.3 | 5.2 | 32.9 |
| blocks.0.fc2 | 6.7 | 5.1 | 13.0 |
| blocks.1.qkv | 3.5 | 5.0 | 81.2 |
| blocks.1.proj | 7.6 | 4.3 | 12.7 |
| blocks.1.fc1 | 3.9 | 4.7 | 30.3 |
| blocks.1.fc2 | 9.6 | 4.9 | 10.6 |
| blocks.2.qkv | 3.5 | 4.4 | 75.3 |
| blocks.2.proj | 3.8 | 4.2 | 10.1 |
| blocks.2.fc1 | 3.7 | 5.7 | 22.3 |
| blocks.2.fc2 | 12.7 | 4.9 | 10.6 |
| blocks.3.qkv | 4.2 | 4.9 | 163.5 |
| blocks.3.proj | 5.3 | 4.4 | 9.3 |
| blocks.3.fc1 | 4.2 | 5.6 | 26.2 |
| blocks.3.fc2 | 14.7 | 5.4 | 8.8 |
