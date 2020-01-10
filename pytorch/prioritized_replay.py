import numpy as np

class SumTree():
    data_pointer = 0 #经验池中新数据替换的位置

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data): 
        tree_index = self.data_pointer + self.capacity - 1 #SumTree中新数据替换的位置
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index-1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):
        '''
        input:
            random sample value
        return:
            SumTree index
            priority value
            transition data 
        '''

        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                break

            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                if self.tree[right_child_index] == 0:
                    v -= (v - self.tree[left_child_index]) # 纠正精度误差
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data = self.data[parent_index - self.capacity + 1]
        return parent_index, self.tree[parent_index], data

    @property
    def total_priority(self):
        return self.tree[0]


class ReplayBuffer():
    PER_e = 0.01 # 0.01 Hyperparameter that we use to avoid some experience to have 0 probability
    PER_a = 0.6 # Hyperparameter that we use to make a tradeoff between only exp with high priority and sampling randomly
    PER_b = 0.4 # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001 #0.001
    abs_error_upper = 2 # 1 clipped abs error
    buffer_size = 0

    def __init__(self, buffer_limit):
        self.sum_tree = SumTree(buffer_limit)

    def put(self, transition):
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        if max_priority == 0:
            max_priority = 1 # 不能让新的transition的priority为0
        self.sum_tree.add(max_priority, transition)
        self.buffer_size += 1
        self.buffer_size = np.minimum(self.buffer_size, self.sum_tree.capacity)

    def sample(self, n):
        mini_batch = []
        b_id = np.empty(n, dtype=np.int32)
        b_ISWeight = np.empty(n, dtype=np.float32)

        priority_segment = self.sum_tree.total_priority / n
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        p_min = np.min(self.sum_tree.tree[-self.sum_tree.capacity:(self.buffer_size+self.sum_tree.capacity-1)]) / self.sum_tree.total_priority
        max_weight = (p_min*self.sum_tree.capacity) ** (-self.PER_b) # 1/n or 1/capacity

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)
            index, priority, data = self.sum_tree.get_leaf(value)

            b_id[i] = index
            # P(i)
            sampling_pro = priority / self.sum_tree.total_priority
            # IS = (1/N * 1/P(i))**b / max wi = (N*P(i))**-b / max wi
            b_ISWeight[i] = (self.sum_tree.capacity * sampling_pro) ** (-self.PER_b) / max_weight  # 1/N = 1/n or 1/capacity
            mini_batch.append(data)

        return b_id, b_ISWeight, np.array(mini_batch)

    def update_batch(self, tree_indexs, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.abs_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_indexs, ps):
            self.sum_tree.update(ti, p)

    def size(self):
        return self.buffer_size

'''
一定要预先填充满整个经验池吗？不用!
但是,经验池的大小一定要是2的次方数！
另外，PER_a要大一点，因为如果不大，优先级之间没有区分度，在抽取的时候，很容易因为精度问题，导致抽取的区间出现偏差。。。。当经验池没有被填满时，这种偏差导致的后果就是选中没被填充的经验池，从而无法选取transition。

可以考虑一开始填满整个经验池

PER_a 太大可能会导致训练过程中出现过拟合。
'''

                