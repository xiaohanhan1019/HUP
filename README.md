### 参数

##### Config.py

- `train_ratio`：train:test=7:3，随机分

- `min_cnt_sku_limit`：去掉数据集中出现次数少于min_cnt_sku_limit的物品

  - 两个数据集都取50

- `min_cnt_line_items`：session长度

  - computers取30
  - appliance取40

- `embedding`大小:

  - 物品(sku)：30

  - 行为(bh)：5

  - 类别(cid3)：8

  - 停留时间(dwell)：5

    - | dwell time bucket | dwell time range |
      | ----------------- | ---------------- |
      | 1                 | 0~15 seconds     |
      | 2                 | 15~40 seconds    |
      | 3                 | 40~100 seconds   |
      | 4                 | 100~600 seconds  |
      | 5                 | 600+ seconds     |

  - 两次click间隔(gap)：5

    - | time interval bucket | time interval range |
      | -------------------- | ------------------- |
      | 1                    | 0~1 seconds         |
      | 2                    | 1~15 seconds        |
      | 3                    | 15~40 seconds       |
      | 4                    | 40~90 seconds       |
      | 5                    | 90+ seconds         |

##### PRNNRec.py

- `layer_num`，RNN的个数？：3
- `rnn_state_size`，RNN隐藏层大小：100，100，100
- `bottom_emb_item_len`，用了多少个embedding（比如SBCGD就是5）：5
- `train_len`，训练集长度：577507(computer)，583282(appliance)
- `test_len`，测试集长度：246967(computer)，250776(appliance)
- `seq_len`，session长度：29(computer)，39(appliance)
- `batch_size`：128
- `epoch`：2
- `drop_out_r`：0
- `loss_weights`，每个loss的权重：0，0.5，0.5
- `learning_rate`：代码里没有设置，他用了RMSprop，应该是使用了keras默认的参数lr=1e-3



### 预处理流程

- 先筛选出现次数符合要求的物品，写入`topsku`，得到`*.topsku`存了筛选物品后的原始数据
  - min_cnt_sku_limit
- 筛选满足session长度的数据 ，得到`*.topsku.len*`
  - min_cnt_line_items和max_cnt_line_items
- 得到`sku.raw`, `gap.raw`, `cid3.raw`, `bh.raw`, `dwell.raw` 以session为单位存了对应的数据
- 去掉session中连续相同的物品，得到`sku.uniq`
  - 以`sku.raw`为输入
  - 但这个去掉连续相同的物品没有意义...后面就没用了
- dwell和gap根据时间长度转换为对应编号，得到`dwell.id`, `gap.id`
- 分别把`item`，`micro behavior`, `category`, `dwell`, `gap`转换为embedding，得到`sku.w2v`, `gap.w2v`, `cid3.w2v`, `bh.w2v`, `dwell.w2v`
  - 不太清楚怎么得到embedding的，具体是通过wordvec.c文件得到的
  - 根据`sku.uniq`得到`sku.w2v`
- 得到reindex以后的各自的embedding和对应的mapping关系，根据`*.w2v`得到
  - 重新对物品编了一下号
  - embedding做了一个归一化
  - 得到各自的`*.reidx`, `*.mapping`
- 得到每个物品最相似的k个物品，得到`top1000sku`
  - 由于内存不够，我直接重写了这个文件`my_top1000.py`，输出和原来一样，只是原来是矩阵计算完一起输出，我是一条一条输出
- 将原始数据根据之前得到的mapping转换，得到`session.SBCGD`
- 把SBCGD整体映射成一个id，得到`session.SBCGD.id`和`session.SBCGD.id.mapping`
- 分成Train和Test，得到`session.SBCGD.train`, `session.SBCGD.test`
  - Train和Test是随机分的，7：3
- 截断train和test
  - **截取了前seq_len_max个**，并且**保证了预测物品和最后一次点击不是一个物品id**
    - 比如 1 2 3 4 5 6 7 8 9 9 9 9 9 -> 1 2 3 4 5 6 7 8 9
  - 得到`session.SBCGD.id.len*.train`,`session.SBCGD.id.len*.test`
- 得到`session.SBCGD.id.len*.train.div`
  - **数据增强？**，前面train只截取了**前seq_len_max**个，这里比如session长度为32，会得到[0,30],[0,31]两条数据
  - 但是`run_PRNN.sh`里面并**没有使用**数据增强后的数据进行训练
- 由`session.SBCGD.id.len*.train`, `session.SBCGD.id.len*.test`, `session.SBCGD.id.len*.train.div` 得到完整的SBCGD数据
  - 会去掉**连续相同**的SBCGD
  - 得到`session.SBCGD.id.len*.SBCGD.train`, `session.SBCGD.id.len*.SBCGD.test`, `session.SBCGD.id.len*.SBCGD.train.div`
- 将SBCGD转换成对应的embedding，会padding，因为之前删掉了连续相同的SBCGD，所以可能会缺失
  - 对SBCGD重新编号，得到reindex后的`session.SBCGD.id.len*.SBCGD.id.train`, `session.SBCGD.id.len*.SBCGD.id.test`, `session.SBCGD.id.len*.SBCGD.id.train.div`
  - 得到SBCGD编号对应的mapping文件`session.SBCGD.id.len30.SBCGD.mapping`和embedding文件`session.SBCGD.id.len30.SBCGD.reidx`
