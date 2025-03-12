## vit

全精度：

```sh
[INFO] load model /home/model/vit/vit-base-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/vit/resultswin-tiny
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.01s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 19.197999954223633, max = 19.517000198364258, mean = 19.305440006256102, median = 19.285999298095703, percentile(99%) = 19.490269775390626
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(19.305440006256102): 51.79887118221294
[INFO] ------------------------------------------------------
```

全量化：

```sh
[INFO] load model /home/model/vit/vit-base-all-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/vit/resultsvit-all
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                                                                                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.87s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 17.659000396728516, max = 18.714000701904297, mean = 17.89604991912842, median = 17.883999824523926, percentile(99%) = 18.235830898284913
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(17.89604991912842): 55.87825271604419
[INFO] ------------------------------------------------------
```

加速比：1.079


