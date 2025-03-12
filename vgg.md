## vgg16

全精度：

```sh
[INFO] load model /home/model/vgg/vgg16-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/vgg/resultvgg16
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.45s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 13.479999542236328, max = 14.32699966430664, mean = 13.685129957199097, median = 13.670999526977539, percentile(99%) = 13.990400457382204
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(13.685129957199097): 73.07201342826471
[INFO] ------------------------------------------------------
```

全量化：

```sh
[INFO] load model /home/model/vgg/vgg16-all-aoe-2-310B1.om success
[INFO] create model description success
[INFO] output path:/home/model/vgg/resultvgg16-all
[INFO] warm up 0 done
Inference array Processing:   0%|                                                                                 | 0/1 [00:00<?, ?it/s]
loop inference exec: (100/100)
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.06it/s]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 7.511000156402588, max = 8.970999717712402, mean = 7.737490000724793, median = 7.724499940872192, percentile(99%) = 8.109699831008916
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(7.737490000724793): 129.24087784363238
[INFO] ------------------------------------------------------
```

加速比：1.769



















