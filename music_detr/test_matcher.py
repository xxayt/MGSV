from matcher import HungarianMatcher
import torch
# 目标张量的格式通常为 [num_target_spans, 2]，这里每个span包含两个坐标值：中心点(c)和宽度(w)
# 这些坐标通常是标准化的，即相对于整体尺寸的比例
targets = [
    {
        "spans": torch.tensor([
            [0.3, 0.1],  # 第一个目标，中心在0.3，宽度为0.1
            [0.7, 0.2]   # 第二个目标，中心在0.7，宽度为0.2
        ])
    }
]

# 假设我们已经有了一些模型输出，我们可以使用这个 targets 列表进行匹配测试
outputs = {
    "pred_logits": torch.tensor([[[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]]),  # 假设有3个查询，每个查询2个类别的概率
    "pred_spans": torch.tensor([[[0.6, 0.15], [0.8, 0.05], [0.2, 0.1]]])  # 3个查询的预测span坐标
}

# 创建匹配器实例
matcher = HungarianMatcher()

# 执行匹配
matched_indices = matcher(outputs, targets)

# 查看匹配结果
print(matched_indices)
# 输出：[(tensor([0, 2]), tensor([1, 0]))]
# 这意味着第0个目标与第2个query的预测匹配，第1个目标与第0个query的预测匹配
# 请注意，这里的索引是从0开始的，因此第一个查询的索引是0，第二个查询的索引是2
# 返回两个张量，第一个张量是预测的索引，第二个张量是目标的索引
