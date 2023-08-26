# ResNet, U-Net 论文阅读成果
## 使用RUTask
### 1. 配置环境
### 2. 输入参数
+ 格式: 
```shell
py ./RUTask.py [-m] <PATH> [-p] <PATH> 
```
+ 参数:
    1. -n: 模型名,必填. 'resnetXX' 'plainnetXX' 'unet' 三选一, 'XX'为数字
    2. -m: 模型路径. 如果路径不存在或未提供会提示是否需要新训练一个模型
    3. -p: predict. 如果task为predict, 需要输入图片路径, 处理后的图像保存到result/img/
+ 示例:
```shell
py ./RUTask.py -n 'resnet18' -m ./result/ckpt/model1.pt -p ./img1.jpg

py ./RUTask.py -n 'unet' -m ./result/ckpt/model2.pt 

py ./RUTask.py -n 'unet' #train a new model
```



#### **施工进度:**
1. unet train部分(现有的脚本能不能用?)
2. Imgnet VOC的数据加载(增强似乎不用)
3. ckpt机制
4. log系统
5. 可视化loss曲线
6. 完善RUT.py
7. 各种bug
8. github推送