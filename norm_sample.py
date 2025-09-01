import numpy as np
from scipy import stats
def sample_within_confidence_interval(means, stds, confidence=0.90, n_samples=1):
    """
    在指定置信区间内采样
    
    参数:
    - means: 均值数组
    - stds: 标准差数组
    - confidence: 置信度 (0-1)
    - n_samples: 每个分布要采样的点数
    
    返回:
    - samples: 采样结果，形状为(n_distributions, n_samples)
    """
    n_distributions = len(means)
    samples = np.zeros((n_distributions, n_samples))
    
    # 计算置信区间边界
    # 对于90%置信区间，z值约为1.645
    z = stats.norm.ppf((1 + confidence) / 2)
    
    for i in range(n_distributions):
        mean = means[i]
        std = stds[i]
        
        # 计算置信区间边界
        lower_bound = mean - z * std
        upper_bound = mean + z * std
        
        # 使用scipy的truncnorm在区间内采样
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        
        samples[i] = stats.truncnorm.rvs(
            a, b, loc=mean, scale=std, size=n_samples
        )
    
    return samples