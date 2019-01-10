import torch
import numpy as np

def py_unique(gt_label):
    unique_labels, unique_id, counts = [],[],[]
    num_id = 0
    cut = 0
    for i in gt_label:
        cut += 1
        if i not in unique_labels:
            unique_labels.append(i)
            num_id += 1
            if num_id != 1:
                counts.append(cut - 1)
                cut = 1
        unique_id.append(num_id - 1)
#        print(i)
    counts.append(cut)
    return unique_labels, unique_id, counts

def unsorted_segment_sum(reshaped_pred, unique_id, num_instances):
    num_len = len(unique_id)
    new_reshaped = []
    reshaped_pred = reshaped_pred.detach().numpy()
    a = reshaped_pred[0]
#    print(type())
    for i in range(num_len):
        if i == 0:
            continue
        if unique_id[i] != unique_id[i - 1]:
            new_reshaped.append(a)
            a = reshaped_pred[i]
        else:
            a = np.add(a, reshaped_pred[i])
    new_reshaped.append(a)
   
    new_reshaped = np.array(new_reshaped)
 
    new_reshaped = torch.from_numpy(new_reshaped).double()

    return new_reshaped

def gather(mu, unique_id):
    mu_expand = []
    for i in unique_id:
        mu_expand.append(mu[i].numpy())
    return mu_expand

def discriminative_loss(
        prediction,
        gt_label,
        feature_dim=1,
        delta_v=20,
        delta_d=40,
        param_var=1.0,
        param_dist=1.0,
        param_reg=1e-3):
    """
    论文equ(1)提到的实例分割损失函数
    :param prediction: inference of network
    :param gt_label: instance label
    :param feature_dim: feature dimension of prediction
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """
    # 像素对齐为一行
    label_shape = gt_label.size()
    gt_label = gt_label.view(label_shape[1] * label_shape[2])
    reshaped_pred =  prediction.view(label_shape[1] * label_shape[2], feature_dim)
    #统计实例个数
    unique_labels, unique_id, counts = py_unique(gt_label)
    # 计算 pixel embedding 均值向量
    counts = np.array(counts)
    counts = torch.from_numpy(counts).double()
    num_instances = len(unique_labels)
#    reshaped_pred = reshaped_pred.numpy()
    segmented_sum = unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    mu = segmented_sum / counts.view(-1, 1)
    mu_expand = gather(mu, unique_id)
    mu_expand = np.array(mu_expand)
    mu_expand = torch.from_numpy(mu_expand)
    
    # 计算公式的 loss(var)
    distance = (mu_expand.double() - reshaped_pred.double()).detach().numpy()
    distance = torch.from_numpy(np.linalg.norm(distance , axis = 1))
    distance -= delta_v
    distance = distance.numpy()
    distance = np.clip(distance, 0., distance)
    distance = np.square(distance)
    distance = torch.from_numpy(distance)
    
    l_var = unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = l_var / counts
    l_var = l_var.numpy()
    l_var = np.sum(l_var)
#    l_var = torch.from_numpy(l_var)
    l_var = l_var / num_instances

#    print('66666')
       
    # 计算公式的 loss （dist）
    mu_interleaved_rep = np.tile(mu.numpy(), (num_instances, 1))
    mu_interleaved_rep = torch.from_numpy(mu_interleaved_rep)
    mu_band_rep = np.tile(mu.numpy(), (1, num_instances))
    mu_band_rep = torch.from_numpy(mu_band_rep)
    mu_band_rep = mu_band_rep.view(num_instances * num_instances, feature_dim)
    mu_diff = mu_band_rep - mu_interleaved_rep

#    print(mu_band_rep.size(), mu_interleaved_rep.size())
    
    # 去除掩模上的零点
    mu_diff = mu_diff.numpy()
    intermediate_tensor = np.sum(np.abs(mu_diff), axis = 1)
    zero_vector = np.zeros(1)
    bool_mask = np.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = mu_diff[bool_mask == True]
    
    mu_norm = np.linalg.norm(mu_diff_bool, axis = 1)
    #mu_norm = torch.from_numpy(mu_norm)
    mu_norm = 2. * delta_d - mu_norm
    mu_norm = np.clip(mu_norm, 0., mu_norm)
    mu_norm = np.square(mu_norm)
    
    l_dist = np.mean(mu_norm)
    # 计算原始 Discriminative Loss 论文中提到的正则项损失
    l_reg = np.mean(np.linalg.norm(mu, axis = 1))
    
    # 合并损失按照原始 Discriminative Loss 论文中提到的参数合并
    from torch.autograd import Variable as V 
    param_scale = V(torch.ones(1), requires_grad = True).double()
    print(param_scale)
    param_var = V(torch.from_numpy(np.array(param_var)), requires_grad = True)
    l_var = torch.from_numpy(np.array(l_var))
    l_var = param_var * l_var
    param_dist = V(torch.from_numpy(np.array(param_dist)), requires_grad = True)
    l_dist = torch.from_numpy(np.array(l_dist))
    l_dist = param_dist * l_dist
    param_reg = V(torch.from_numpy(np.array(param_reg)), requires_grad = True)
    l_reg = torch.from_numpy(np.array(l_reg))
    l_reg = param_reg * l_reg
    loss = param_scale * (l_var + l_dist + l_reg)
#    loss = torch.from_numpy(np.array(loss)) 
#    print(loss, l_var, l_dist, l_reg) 
 
    return loss


if __name__ == '__main__':
    bsize, fdim = 1, 2
    pred = torch.FloatTensor(bsize, fdim, 12, 32)
    gt = torch.FloatTensor(bsize, 12, 32)
#    print('88888')
    output = discriminative_loss(pred, gt, feature_dim=fdim)
    output.backward()
    print(output)
