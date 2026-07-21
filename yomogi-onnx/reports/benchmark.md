# Yomogi ONNX benchmark

CPU, intra-op 1, inter-op 1, warmup 20, measurement 200; each variant ran in a fresh process.

| Variant | Chars | Median ms | p95 ms | p99 ms | chars/s | Load s | Peak RSS MiB | Artifacts MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch_cpu | 10 | 29.678 | 35.101 | 37.392 | 336.9 | 4.509 | 818.6 | 93.7 |
| pytorch_cpu | 30 | 53.216 | 57.246 | 60.447 | 563.7 | 4.509 | 818.6 | 93.7 |
| pytorch_cpu | 100 | 157.502 | 172.572 | 247.434 | 634.9 | 4.509 | 818.6 | 93.7 |
| pytorch_cpu | 300 | 408.705 | 422.178 | 439.300 | 734.0 | 4.509 | 818.6 | 93.7 |
| encoder_fp32_memory | 10 | 2.261 | 2.809 | 3.110 | 4422.3 | 1.839 | 436.8 | 96.2 |
| encoder_fp32_memory | 30 | 38.947 | 43.939 | 45.662 | 770.3 | 1.839 | 436.8 | 96.2 |
| encoder_fp32_memory | 100 | 131.098 | 136.107 | 137.599 | 762.8 | 1.839 | 436.8 | 96.2 |
| encoder_fp32_memory | 300 | 389.614 | 403.607 | 518.560 | 770.0 | 1.839 | 436.8 | 96.2 |
| encoder_fp32_mmap | 10 | 2.255 | 2.789 | 2.987 | 4434.6 | 1.868 | 394.2 | 96.2 |
| encoder_fp32_mmap | 30 | 40.005 | 43.310 | 44.560 | 749.9 | 1.868 | 394.2 | 96.2 |
| encoder_fp32_mmap | 100 | 133.669 | 138.093 | 140.739 | 748.1 | 1.868 | 394.2 | 96.2 |
| encoder_fp32_mmap | 300 | 389.281 | 402.940 | 406.182 | 770.7 | 1.868 | 394.2 | 96.2 |
| full_fp32 | 10 | 2.240 | 2.972 | 5.750 | 4463.9 | 1.913 | 436.6 | 96.2 |
| full_fp32 | 30 | 39.611 | 43.619 | 114.548 | 757.4 | 1.913 | 436.6 | 96.2 |
| full_fp32 | 100 | 131.170 | 135.464 | 138.965 | 762.4 | 1.913 | 436.6 | 96.2 |
| full_fp32 | 300 | 386.917 | 399.771 | 408.444 | 775.4 | 1.913 | 436.6 | 96.2 |
| encoder_int8_experimental | 10 | 0.570 | 0.727 | 0.891 | 17553.1 | 1.856 | 422.1 | 80.8 |
| encoder_int8_experimental | 30 | 1.387 | 1.681 | 1.870 | 21634.1 | 1.856 | 422.1 | 80.8 |
| encoder_int8_experimental | 100 | 6.742 | 49.552 | 51.334 | 14832.3 | 1.856 | 422.1 | 80.8 |
| encoder_int8_experimental | 300 | 140.785 | 188.550 | 208.867 | 2130.9 | 1.856 | 422.1 | 80.8 |

RSS is process RSS, not a model-exclusive allocation. Artifact size includes the files required by that variant.
