import numpy as np
from tqdm import tqdm

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)  # Cộng một lượng bất kỳ

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))    # nhân tử bất kì lấy theo phân phối chuẩn với sigma = 0.1 là độ lệch chuẩn
    return np.multiply(x, factor[:,np.newaxis,:])  

def rotation(x):    # flip lộn hết lên
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2])) # x.shape[0] là số chuỗi thời gian, x.shape[2] là số chiều
    rotate_axis = np.arange(x.shape[2]) # x.shape[2] là số chiều
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]    # nhân 1 hoặc -1 với giá trị => lật

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])  # x.shape[1] là số mốc thời gian, số bước. orig_steps = [ 0, 1, 2, 3, ..., x.shape[1] - 1]
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))    # num_segs là mảng cỡ shape[0] nghĩa là có cỡ cùng với số chuỗi thời gian và lấy giá trị bất kỳ trong đoạn [1, max_segments]
    
    ret = np.zeros_like(x)  # Tạo mảng cùng cấu trúc với x và khởi tạo các giá trị là 0
    for i, pat in enumerate(x): # Duyệt qua các phần tử của x. với x là tập các chuỗi thời gian thì pat là các chuỗi thời gian
        if num_segs[i] > 1: # Nếu cỡ của x > 1 (x có tồn tại ít nhất một phần tử chuỗi thời gian)
            if seg_mode == "random":    # Nếu seg_mode truyền vào "random"
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False) # split_points Lấy ngẫu nhiên một đoạn dài num_segs[i] - 1 điểm
                split_points.sort() # sắp xếp lại đoạn vừa lấy ngẫu nhiên ấy
                splits = np.split(orig_steps, split_points)     # Chia mảng ban đầu thành các phần nhỏ một cách ngẫu nhiên
            else:
                splits = np.array_split(orig_steps, num_segs[i])    # chia mảng ban đầu thành các phần nhỏ
            warp = np.concatenate(np.random.permutation(splits)).ravel()       # ghép các mảng đó vào với nhau
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    '''
    Hàm này tạo random một loạt các giá trị ứng với các mốc thời điểm của chuỗi thời gian
    Chia chuỗi thời gian thành knot + 2 phần bằng nhau

    '''
    from scipy.interpolate import CubicSpline   # Hàm nội suy bậc 3
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))    # -> y
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T            # -> x = [0, 25, 50, 76, 101, 127]
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # chuỗi thời gian i -> chuỗi random thứ i
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper   # mỗi pat ứng với một chuỗi thời gian, nhân thêm một cái warper nữa

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline   # Hàm nội suy bậc 3
    orig_steps = np.arange(x.shape[1])          # Các bước thời gian, [0, 1, .., n]
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:    # nếu độ dài "cửa sổ" > độ dài của chuỗi thời gian thì trả về luôn chuỗi thời gian
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)    # bắt đầu tại điểm bất kì
    ends = (target_len + starts).astype(int)                                     # kết thúc tại bắt đầu + độ dài khung cửa sổ
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x): # Sinh dữ liệu từ chuỗi thời gian thứ i, pat là chuỗi thời gian
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import utils.dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])    # Danh sách x.shape[0] mốc thời gian bất kì (x.shape[0] tương ứng với số chuỗi thời gian trong bộ dữ liệu)
    window = np.ceil(x.shape[1] / 10.).astype(int)  # T / 10
    orig_steps = np.arange(x.shape[1])  # [1, 2, 3, ..., T]
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels    # Không rõ lắm nhưng trường hợp nhãn 1 chiều thì l = labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):   # Lặp qua các chuỗi thời gian.
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)   # Bỏ chuỗi thời gian đó khỏi danh sách lựa chọn
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]   # Bỏ các chuỗi thời gian không cùng lớp
        if choices.size > 0:      # Nếu choices tồn tại ít nhất một phần tử
            random_sample = x[np.random.choice(choices)]    # Lấy ra một chuỗi thời gian bất kì từ choices
            # SPAWNER splits the path into two randomly
            # dtw hai chuỗi thời gian: pat và random_sample, chia 2 chuỗi này bằng điểm thời gian ngẫu nhiên random_points[i].
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            # Ghép hai path lại
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, dtw.RETURN_ALL, window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            # Tính ma trận trung bình
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                # nội suy
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:       # không còn gì trong choices sau khi lọc bỏ chuỗi thời gian không cùng lớp với chuỗi thời gian được chọn
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])   # Hiển thị thông báo chỉ có một mẫu của lớp l
            ret[i,:] = pat  # gán chuỗi thời gian thứ i mới bằng chuỗi thời gian ban đầu
    return jitter(ret, sigma=sigma) # khi trả về chạy jittering để trộn lại mẫu một lần nữa



    

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)  # 1 window = 10% khung thời gian (T/10)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels    # danh sách các nhãn/lớp
        
    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])): # duyệt lần lượt các chuỗi thời gian trong bộ dữ liệu
        # get the same class as i
        choices = np.where(l == l[i])[0]    # choices là danh sách các chuỗi thời gian cùng nhãn/lớp với chuỗi hiện tại
        if choices.size > 0:                # Nếu tồn tại ít nhất một chuỗi thời gian cùng lớp với chuỗi đang duyệt
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]  # Lấy k chuỗi thời gian ra
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))   # khởi tạo ma trận k x k
            for p, prototype in enumerate(random_prototypes):   # Duyệt lần lượt k chuỗi thời gian
                for s, sample in enumerate(random_prototypes):      # duyệt qua lần nữa k chuỗi thời gian đó
                    if p == s:  # Nếu prototype trùng sample (prototype index và sample index trùng nhau) nói cách khác cùng một chuỗi thời gian
                        dtw_matrix[p, s] = 0.
                    else:   # Nếu không thì tính khoảng cách căn chỉnh giữa hai chuỗi thời gian
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]   # lấy id chuỗi thời gian
            nearest_order = np.argsort(dtw_matrix[medoid_id])   # sắp xếp dtw_matrix[medoid_id]
            medoid_pattern = random_prototypes[medoid_id]   # Lấy ra một chuỗi thời gian từ random_prototypes
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)     # Khởi tạo một chuỗi thời gian
            weighted_sums = np.zeros((medoid_pattern.shape[0])) # khởi tạo mảng T phần tử 0 (T là độ dài các chuỗi thời gian)
            for nid in nearest_order:      
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]    
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])  # w_i
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:   # Nếu không có mẫu thứ hai cùng lớp/nhãn
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i] # 
    return ret

# Proposed

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]   # path[1] ứng với random_prototype
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)  # 1 window = 10% độ dài chuỗi thời gian
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])

    for i, pat in enumerate(tqdm(x)):   # duyệt qua các chuỗi thời gian
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)   # Bỏ chuỗi thời gian hiện tại

        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]  # Bỏ các chuỗi thời gian khác lớp
        negative = np.where(l[choices] != l[i])[0]  # Bỏ các chuỗi thời gian cùng lớp

        if positive.size > 0 and negative.size > 0:     # Nếu tồn tại cả positive và negative
            pos_k = min(positive.size, positive_batch)  
            neg_k = min(negative.size, negative_batch)  
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]   # tập các mẫu cùng lớp
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]   # tập các mẫu khác lớp
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)    # Chọn chuỗi thời gian có khoảng cách giữa neg_aves và pos_aves lới nhất
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return ret

def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")
