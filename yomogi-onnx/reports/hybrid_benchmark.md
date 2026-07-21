# Yomogi + Kanalizer hybrid benchmark

| case | API | cache | median ms | p95 ms | Yomogi ms | Kanalizer ms | total ms |
|---|---|---|---:|---:|---:|---:|---:|
| japanese_10 | sync | warm | 2.460 | 2.826 | 2.331 | 0.000 | 2.454 |
| japanese_10 | async | warm | 2.543 | 3.021 | 2.273 | 0.000 | 2.539 |
| japanese_10 | sync | cold | 2.414 | 2.948 | 2.300 | 0.000 | 2.408 |
| japanese_10 | async | cold | 2.528 | 3.392 | 2.260 | 0.000 | 2.525 |
| english_1 | sync | warm | 3.263 | 3.736 | 3.113 | 0.014 | 3.255 |
| english_1 | async | warm | 3.445 | 3.934 | 3.176 | 0.013 | 3.442 |
| english_1 | sync | cold | 3.934 | 4.373 | 3.812 | 3.459 | 3.927 |
| english_1 | async | cold | 4.032 | 4.456 | 3.761 | 3.279 | 4.028 |
| english_3 | sync | warm | 17.789 | 43.980 | 17.425 | 0.029 | 17.767 |
| english_3 | async | warm | 24.177 | 42.829 | 23.257 | 0.068 | 24.171 |
| english_3 | sync | cold | 70.068 | 72.982 | 67.899 | 68.147 | 70.026 |
| english_3 | async | cold | 62.206 | 72.169 | 56.704 | 61.074 | 62.199 |
| english_repeat | sync | warm | 32.842 | 33.957 | 32.022 | 0.065 | 32.808 |
| english_repeat | async | warm | 29.662 | 37.666 | 28.930 | 0.060 | 29.657 |
| english_repeat | sync | cold | 31.226 | 35.263 | 30.506 | 25.242 | 31.190 |
| english_repeat | async | cold | 28.209 | 32.362 | 27.599 | 23.684 | 28.202 |
