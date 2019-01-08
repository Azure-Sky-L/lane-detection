import torch as t
import numpy as np

def py_unique(correct_label):
    unique_labels, unique_id, counts = [],[],[]
    num_id = 0
    cut = 0
    for i in correct_label:
        cut += 1
        if i not in unique_labels:
            unique_labels.append(i)
            num_id += 1
            if num_id != 1:
                counts.append(cut - 1)
                cut = 1
        unique_id.append(num_id - 1)
    counts.append(cut)
    return unique_labels, unique_id, counts

def unsorted_segment_sum(reshaped_pred, unique_id, num_instances):
    num_len = len(unique_id)
    new_reshaped = []
    a = reshaped_pred[0]
    for i in range(num_len):
        if i == 0:
            continue
        if unique_id[i] != unique_id[i - 1]:
            new_reshaped.append(a)
            a = reshaped_pred[i]
        else:
            a.add_(reshaped_pred[i])
    new_reshaped.append(a)
    
    new_reshaped = t.from_numpy(np.array(new_reshaped))
    return new_reshaped

def gather(mu, unique_id):
    mu_expand = []
    for i in unique_id:
        mu_expand.append(mu[i])
    return mu_expand

def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    """
    论文equ(1)提到的实例分割损失函数
    :param prediction: inference of network
    :param correct_label: instance label
    :param feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """
    # 像素对齐为一行
    correct_label = correct_label.view(label_shape[1] * label_shape[0])
    reshaped_pred = reshaped_prd.view(label_shape[1] * label_shape[0], feature_dim)
    
    #统计实例个数
    unique_labels, unique_id, counts = py_unique(correct_label)
    # 计算 pixel embedding 均值向量
    counts = t.from_numpy(counts).float()
    num_instances = len(unique_labels)
    segmented_sum = unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    mu = segmented_sum / counts.view(-1, 1)
    mu_expand = gather(mu, unique_id)
    
    # 计算公式的 loss(var)
    distance = t.from_numpy(np.linalg.norm((mu_expand - reshaped_pred).numpy(), axis = 1))
    distance -= delta_v
    distance = distance.numpy()
    distance = np.clip(distance, 0., distance)
    distance = np.square(distance)
    distance = t.from_numpy(distance)
    
    l_var = unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = l_var / counts
    l_var = l_var.numpy()
    l_var = np.sum(l_var)
    l_var = t.from_numpy(l_var)
    l_var = l_var // num_instances.float()
    
    # 计算公式的 loss （dist）
    mu_interleaved_rep = np.tile(mu.numpy(), (num_instances, 1))
    mu_interleaved_rep = t.from_numpy(mu_interleaved_rep)
    mu_band_rep = np.tile(mu.numpy(), (1, num_instances))
    mu_band_rep = t.from_numpy(mu_band_rep)
    mu_band_rep.view(num_instances * num_instances, feature_dim)
    mu_diff = mu_band_rep - mu_interleaved_rep
    
    # 去除掩模上的零点
    mu_diff = mu_diff.numpy()
    intermediate_tensor = np.sum(np.abs(mu_diff), axis = 1)
    zero_vector = np.zeros(1)
    bool_mask = np.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = mu_diff[bool_mask == True]
    
    mu_norm = np.norm(mu_diff_bool, axis = 1)
    #mu_norm = t.from_numpy(mu_norm)
    mu_norm = 2. * delta_d.numpy() - mu_norm
    mu_norm = np.clip(mu_norm, 0., mu_norm)
    mu_norm = np.square(mu_norm)
    
    # 计算原始 Discriminative Loss 论文中提到的正则项损失
    l_reg = np.mean(np.norm(mu, axis = 1))
    
    # 合并损失按照原始 Discriminative Loss 论文中提到的参数合并
    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg
    loss = param_scale * (l_var + l_dist + l_reg)
    
    return loss, l_var, l_dist, l_reg
