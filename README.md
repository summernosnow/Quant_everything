# Yolov10s
| Number |                             Type                             |             Value             |
| :----: | :----------------------------------------------------------: | :---------------------------: |
|   1    |           pth Accuracy (%) of full precision model           |             46.3%             |
|   2    |          onnx Accuracy (%) of full precision model           |             44.3%             |
|   3    |    onnx Accuracy (%) of full quantized model (e.g., w8a8)    |             42.4%             |
|   4    | onnx Accuracy (%) of mixed quantization model by our quantizing framework |            44.18%             |
|   5    |    onnx Accuracy (%) of AMCT-like addressed model from 4     |            44.12%             |
|   6    |        Accuracy (%) of om-like addressed model from 5        |             45.6%             |
|   7    |        Accuracy (%) of om-like addressed model from 2        |             46.0%             |
|   8    |        Accuracy (%) of om-like addressed model from 3        |             44.5%             |
|   9    |        Latency (ms) of om-like-formatted model from 5        |        13.18793509 ms         |
|   10   |        Latency (ms) of om-like-formatted model from 2        | 15.64537856 ms/13.25025012 ms |
|   11   |        Latency (ms) of om-like-formatted model from 3        |        13.07029841 ms         |
|   o0   |                        diff(4,5)<0.1%                        |             True              |
|  oa1   |    maximal simulated accuracy (%) drop, i.e., diff(2, 4)     |             0.12%             |
|  oa2   |    simulated accuracy (%) drop of ours, i.e., diff(2, 3)     |             1.9%              |
|  oa3   |    maximal practical accuracy (%) drop, i.e., diff(7, 8)     |             1.5%              |
|  oa4   |    practical accuracy (%) drop of ours, i.e., diff(7, 6)     |             0.4%              |
|  os1   |              maximal speedup, i.e., div(10, 11)              |          1.197/1.014          |
|  os2   |             speedup (%) of ours, i.e., div(10,9)             |          1.186/1.005          |
