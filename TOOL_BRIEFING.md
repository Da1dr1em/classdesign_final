# Python 噪声生成工具介绍

## 工作流程执行顺序

### 模块2 - 噪声生成策略 (class_ class_1... class_6)

执行顺序:

1. **配置表加载**
   - 加载 alphabet_config.mat
   - 3行3列权重配置矩阵
   - 创建噪声类型适配器

2. **模块列生成 (模块1→模块2)**
   - 参数列适配 (snr_low_sn, param_6_sn)
   - 类型列表选择 (type_value_sn 6个选项)
   - 创建完整的列序列

3. **参数序列生成及封装**
   - 序列分段和封装
   - 检查逻辑常量和异常处理 (try/except)
   - 封装形成 sn_params_matrix 数据结构

4. **噪声生成执行器 (run_sn_gen)**
   - 循环生成点坐标矩阵 sn_px_sn
   - 在点处生成对应信噪比 (sn_sn 9-17db)
   - 计算幅度矩阵 sn_val_sn
   - sn_counter = sn_value_sn
   公式:

   ```
   sn_random_val_matrix = np.random.randint(0, 3, size=(3,3))
   sn_value_sn_mtx = np.multiply(sn_unit_val, sn_random_val_matrix)
   sn_sn_params_matrix = sn_counter * sn_value_counter * sn_sn_params_matrix
   ```

5. **结果文件生成**
   - 输出形式取决于SN值参数 (0.5或4.96)
   - 结合db_sn输出配置矩阵 sn_mtx_sn
   - 分类器结果决定输出内容 (snr类-策略选择)

6. 默认适配策略:
   - 当db_count和sn_count满足条件时启用
   - 存储路径: data/noise/
   - 错误机制: 3列错误权重矩阵

当db_count ≥ 5 或 sn_count +1 时:
   - 启用噪音写入标记 (db_noisemask_type_voice=2)
   - db_sn_matrix = sn_random_val_matrix

最后输出:

```
写噪完成 -> 执行标记 sn_e2_processing=True
```