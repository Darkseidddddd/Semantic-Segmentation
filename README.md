### <center/> 实验报告-BiSeNet <center/>

---

#### <center/> 实验一<center/>

#### 一. 数据集

Potsdam RGB 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

这里把训练集，测试集按6 : 2 : 2分成训练集、验证集和测试集，此时训练集有3288张图片，验证集有1092张图篇，测试集有1092张图片。

#### 二. 实验参数

> backbone：resnet18
>
> 1080 8G × 1
>
> epochs：1000
>
> batch_size：2
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：99h

其中学习率按照如下公式进行衰减

```python
lr = init_lr*(1 - iter/max_iter)**power
```

其中`init_lr`为初始学习率，`iter`为迭代次数，`max_iter`为总的训练迭代次数，`power`为指数，这里设为0.9。

#### 三. 实验结果

##### （1）train_loss

<img src="images\1_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）val_miou

<img src="images\1_epoch_miou_val.png" style="zoom:67%;" />

最高到***0.8004***。

##### （3）val_oa

<img src="images\1_epoch_precision_val.png" style="zoom:67%;" />

最高到***0.8731***。

##### （4）test

> IoU for each class:
>
> surfaces:0.824,
>
> Building:0.886,
>
> LowVegetation:0.735,
>
> Tree:0.740,
>
> Car:0.819,
>
> Clutter:0.424,
>
> oa for test: 0.877
>
> mIoU: 0.738

测试集miou为***0.738***，oa为***0.877***。

#### <center/> 实验二 <center/>

#### 一. 数据集

Potsdam RGB 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

这里把训练集，测试集按6 : 2 : 2分成训练集、验证集和测试集，此时训练集有3288张图片，验证集有1092张图篇，测试集有1092张图片。

#### 二. 实验参数

> backbone：resnet18
>
> 1080 8G × 1
>
> epochs：1000
>
> batch_size：2
>
> 初始lr：0.025
>
> optimizer：Adadelta
>
> 训练时长：99h

其中学习率按照如下公式进行衰减

```python
lr = init_lr*(1 - iter/max_iter)**power
```

其中`init_lr`为初始学习率，`iter`为迭代次数，`max_iter`为总的训练迭代次数，`power`为指数，这里设为0.9。

#### 三. 实验结果

##### （1）train_loss

<img src="images\3_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\3_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\3_epoch_OA.png" style="zoom:67%;" />

##### （4）test

> IoU for each class:
>
> surfaces:0.824,
>
> Building:0.896,
>
> LowVegetation:0.734,
>
> Tree:0.758,
>
> Car:0.825,
>
> Clutter:0.350,
>
> oa for test: 0.880
>
> mIoU: 0.731

测试集miou为***0.731***，oa为***0.88***。

#### <center/> 实验三 <center/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet18
>
> 1080 8G × 1
>
> epochs：1000
>
> batch_size：2
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：100+h

其中学习率按照如下公式进行衰减

```python
lr = init_lr*(1 - iter/max_iter)**power
```

其中`init_lr`为初始学习率，`iter`为迭代次数，`max_iter`为总的训练迭代次数，`power`为指数，这里设为0.9。

#### 三. 实验结果

实验中途断过几次，这是600到1000个epochs的情况。

##### （1）train_loss

<img src="images\2_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\2_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\2_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test

> IoU for each class:
>
> surfaces:0.800
>
> Building:0.882
>
> LowVegetation:0.694
>
> Tree:0.718
>
> Car:0.810
>
> Clutter:0.357
>
> oa for test: 0.863
>
> mIoU for test: 0.711

测试集miou为***0.711***，oa为***0.863***。

#### <center/> 实验四 <center/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet101
>
> Titan Xp 12G × 1
>
> epochs：1000
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：296h

#### 三. 实验结果

##### （1）train_loss

<img src="images\4_epoch_loss_epoch_train.png" style="zoom:67%;" />

1000个epochs并没有收敛。

##### （2）train_miou

<img src="images\4_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\4_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test_miou

<img src="images\4_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.808***，epochs为***991***，此时对应oa为***0.8799***。

##### （5）test_oa

<img src="images\4_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.88***，epochs为***966***，此时对应的miou为***0.8069***。

训练集loss曲线没有收敛，测试集的miou和oa都还有上升的空间。

#### <center/> 实验五 <center/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet50 
>
> 1080 8G × 2
>
> epochs：1000
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：321h

#### 三. 实验结果

##### （1）train_loss

<img src="images\5_epoch_loss_epoch_train.png" style="zoom:67%;" />

可以看到1000个epochs还是没有收敛。

##### （2）train_miou

<img src="images\5_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\5_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test_miou

<img src="images\5_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.8044***，epochs为***993***，此时对应的oa为***0.878***。

##### （5）test_oa

<img src="images\5_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.878***，epochs为***993***，此时miou为***0.8044***。

训练集上loss曲线没有收敛，测试集的miou和oa都还有上升的空间。

#### <center/> 实验六 <center/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet50 
>
> 1080 8G × 2
>
> epochs：559
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：adadelta
>
> 训练时长：162h

#### 三. 实验结果

##### （1）train_loss

<img src="images\6_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\6_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\6_epoch_oa_train.png" alt="epoch_oa_train" style="zoom:67%;" />

##### （4）test_miou

<img src="images\6_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.8087***，epochs为***414***，此时对应的oa为***0.88***。

##### （5）test_oa

<img src="images\6_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.88***，epochs为***414***，此时miou为***0.8087***。

#### <center/> 实验七 <center/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet18 
>
> 1080 8G × 2
>
> epochs：571
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：adadelta
>
> 训练时长：130h

#### 三. 实验结果

##### （1）train_loss

<img src="images\7_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\7_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\7_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test_miou

<img src="images\7_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.7984***，epochs为***517***，此时对应的oa为***0.8751***。

##### （5）test_oa

<img src="images\7_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.876***，epochs为***499***，此时miou为***0.7976***。

### <center/> FastFCN <center/>

---

#### <center/> 实验一 <cetner/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet50 
>
> 1080 8G × 2
>
> epochs：1000
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：182h
>
> 无jpu模块

#### 三. 实验结果

##### （1）train_loss

<img src="images\8_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\8_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\8_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test_miou

<img src="images\8_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.6551***，epochs为***976***，此时对应的oa为***0.8485***。

##### （5）test_oa

<img src="images\8_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.8495***，epochs为***966***，此时miou为***0.6525***。

很明显没有收敛。

#### <center/> 实验二 <cetner/>

#### 一. 数据集

Potsdam IRRG 三通道

总共六分类

> surfaces  (RGB: 255, 255, 255) 
>
> Building  (RGB: 0, 0, 255) 
>
> Low vegetation  (RGB: 0, 255, 255) 
>
> Tree  (RGB: 0, 255, 0) 
>
> Car  (RGB: 255, 255, 0) 
>
> Clutter / background  (RGB: 255, 0, 0) 

训练集总共24张6000 × 6000 × 3的图片，测试集总共14张6000 × 6000 × 3的图片

把所有训练集和测试集切割为512 × 512 × 3的图片，对于每行每列最后一张除不尽的部分，用上一张图片最后的部分像素填充至512 × 512大小。

切割后的训练集有3456张图片，测试集有2016张图片，大小为512 × 512 × 3。

#### 二. 实验参数

> backbone：resnet50 
>
> 1080 8G × 2
>
> epochs：1000
>
> batch_size：6
>
> 初始lr：0.025
>
> optimizer：sgd
>
> 训练时长：228h
>
> 有jpu模块

#### 三. 实验结果

##### （1）train_loss

<img src="images\9_epoch_loss_epoch_train.png" style="zoom:67%;" />

##### （2）train_miou

<img src="images\9_epoch_miou_train.png" style="zoom:67%;" />

##### （3）train_oa

<img src="images\9_epoch_oa_train.png" style="zoom:67%;" />

##### （4）test_miou

<img src="images\9_epoch_miou_test.png" style="zoom:67%;" />

最高到***0.7124***，epochs为***979***，此时对应的oa为***0.8682***。

##### （5）test_oa

<img src="images\9_epoch_oa_test.png" style="zoom:67%;" />

最高到***0.8688***，epochs为***996***，此时miou为***0.7111***。

#### 总结

##### <center/> BiSeNet <center/>

---

| 实验   | batch_size/backbone | 显卡     | 数据集 | optimizer | miou/step  | oa/step   |
| ------ | ------------------- | -------- | ------ | --------- | ---------- | --------- |
| 实验一 | 2/resnet18          | 1080 × 1 | RGB    | sgd       | 0.738/?    | 0.877/?   |
| 实验二 | 2/resnet18          | 1080 × 1 | RGB    | adadelta  | 0.731/?    | 0.88/?    |
| 实验三 | 2/resnet18          | 1080 × 1 | IRRG   | sgd       | 0.711/?    | 0.863/?   |
| 实验四 | 6/resnet101         | Titan Xp | IRRG   | sgd       | 0.808/991  | 0.88/966  |
| 实验五 | 6/resnet50          | 1080 × 2 | IRRG   | sgd       | 0.8044/993 | 0.878/993 |
| 实验六 | 6/resnet50          | 1080 × 2 | IRRG   | adadelta  | 0.8087/414 | 0.88/414  |
| 实验七 | 6/resnet18          | Titan Xp | IRRG   | adadelta  | 0.7984/517 | 0.876/499 |

| batch_size/backbone | 显卡     | optimizer | miou/step  | oa/step   | F-score | kappa |
| ------------------- | -------- | --------- | ---------- | --------- | ------- | ----- |
| 6/resnet50          | 1080 × 2 | sgd       | 0.8044/993 | 0.878/993 | 0.8856  | 0.774 |
| 6/resnet101         | Titan Xp | sgd       | 0.808/991  | 0.88/966  | 0.89    | 0.796 |
| 6/resnet18          | Titan Xp | adadelta  | 0.7984/517 | 0.876/499 | 0.8756  | 0.771 |
| 6/resnet50          | 1080 × 2 | adadelta  | 0.8087/414 | 0.88/414  | 0.893   | 0.803 |

512 × 512 × 3，FPS如下

|           | 模型大小 | 1080 8G × 1 | Titan Xp 12G × 1 |
| --------- | -------- | ----------- | ---------------- |
| resnet18  | 47.6M    | 183.3       | 196.6            |
| resnet50  | 120.1M   | 97.1        | 105.2            |
| resnet101 | 192.8M   | 56          | 61.8             |

##### <center/> FastFCN <center/>

---

| 实验   | batch_size/backbone | 显卡     | jpu模块 | optimizer | miou/step  | oa/step   |
| ------ | ------------------- | -------- | ------- | --------- | ---------- | --------- |
| 实验一 | 6/resnet50          | 1080 × 2 | 无      | sgd       | 0.6551/976 | 0.85/966  |
| 实验二 | 6/resnet50          | 1080 × 2 | 有      | sgd       | 0.7124/979 | 0.869/996 |

512 × 512 × 3，FPS如下

|          | 模型大小 | jpu模块 | 1080 8G × 1 | Titan Xp 12G × 1 |
| -------- | -------- | ------- | ----------- | ---------------- |
| resnet50 | 120M     | 无      | 59.6        | -                |
| resnet50 | 179M     | 有      | 26.5        | -                |

可视化结果：

实验六的`confusion matrix`

<img src="igarss-bisenet\confusion_matrix.png" style="zoom: 50%;" />

<img src="igarss-bisenet\loss.png" alt="loss" style="zoom: 67%;" />

<img src="igarss-bisenet\miou.png" alt="miou" style="zoom: 67%;" />

<img src="igarss-bisenet\oa.png" alt="oa" style="zoom: 67%;" />