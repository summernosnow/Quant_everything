## swin-tiny

全精度：

```sh
[INFO] load model /home/model/swin/swin-tiny-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/swin/resultswin-tiny
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.35s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 11.960000038146973, max = 14.02299976348877, mean = 12.25078999519348, median = 12.176000118255615, percentile(99%) = 13.876479997634888
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(12.25078999519348): 81.62738895959718
[INFO] ------------------------------------------------------
```

全量化：

```sh
[INFO] load model /home/model/swin/swin-tiny-all-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/swin/resultswin-tiny-all
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.78s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 16.733999252319336, max = 17.625, mean = 17.039719944000243, median = 17.045000076293945, percentile(99%) = 17.374529743194582
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(17.039719944000243): 58.68641053294448
[INFO] ------------------------------------------------------
```

加速比：0.719
