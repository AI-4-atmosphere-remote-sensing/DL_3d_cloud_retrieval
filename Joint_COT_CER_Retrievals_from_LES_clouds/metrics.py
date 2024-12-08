import numpy as np
import torch
np.random.seed(0)
def compute_rmse(y_hat,y):
    ret=0
    if isinstance(y_hat, torch.Tensor):
        ret=torch.sqrt(((y_hat - y)**2).mean())
    elif isinstance(y_hat, np.ndarray):
        ret=np.sqrt(((y_hat - y)**2).mean())
    return ret

def masked_rmse(y_hat,y, mask):
    total_elements=0
    if isinstance(mask, torch.Tensor):
        # total_elements = torch.numel(mask)
        total_elements = torch.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    else:
        # total_elements = mask.size
        total_elements = np.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    ret=0
    # print("types of input: ",type(y_hat),type(y))
    if total_elements==0:
        ret=0
    elif isinstance(y_hat, torch.Tensor):
        ret=torch.sqrt(torch.sum((y_hat-y)**2)/total_elements)
    else: 
        mse = np.sum(np.square(y_hat-y))
        ret=np.sqrt(mse/total_elements)
        # print("Computing Loss: ",mse)
    return ret

def masked_mse(y_hat,y, mask):
    total_elements=0
    if isinstance(mask, torch.Tensor):
        # total_elements = torch.numel(mask)
        total_elements = torch.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    else:
        # total_elements = mask.size
        total_elements = np.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    ret=0
    if total_elements==0:
        ret = 0
    elif isinstance(y_hat, torch.Tensor):
        ret=torch.sum((y_hat-y)**2)/total_elements
    else:
        ret=np.sum((y_hat-y)**2)/total_elements
    return ret

def masked_mae(y_hat,y, mask):
    total_elements=0
    if isinstance(mask, torch.Tensor):
        # total_elements = torch.numel(mask)
        total_elements = torch.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    else:
        # total_elements = mask.size
        total_elements = np.sum(mask)
        y_hat = y_hat*mask
        y     = y*mask
    ret=0
    if total_elements==0:
        ret = 0
    elif isinstance(y_hat, torch.Tensor):
        ret=torch.sum(torch.abs(y_hat-y))/total_elements
    else:
        ret=np.sum(np.abs(y_hat-y))/total_elements
    return ret


def rrmse(y_hat,y):
    ret=0
    det=1
    if isinstance(y_hat, torch.Tensor):
        ret=torch.sqrt(((y_hat - y)**2).mean())
        det= y.mean()

    elif isinstance(y_hat, np.ndarray):
        ret=np.sqrt(((y_hat - y)**2).mean())
        det= y.mean()
    return ret/det

def dice_score(y_hat,y,smooth=0.01):
    assert type(y_hat)==type(y)
    if isinstance(y_hat, torch.Tensor):
        cmn= torch.sum(y_hat*y)
        denom= torch.sum(y_hat)+torch.sum(y)
        if denom==0:
            return torch.tensor(0)

    elif isinstance(y_hat, np.ndarray):
        cmn=np.sum(y_hat*y)
        denom= np.sum(y_hat)+np.sum(y)
        if denom==0:
            return 0
    return (2*cmn+smooth)/(denom+smooth)

def compute_dice_score(y_pred,y_true,L=10,Lmin=-6.6226,Lmax=9.3373):
    history=[]
    score= []
    delta = (Lmax-Lmin)/L
    assert type(y_pred)==type(y_true)

    if isinstance(y_pred, torch.Tensor):
        # create color bins and compute dice scores for each bin
        for l in range(L):
            lower_limit = Lmin+delta*l
            upper_limit = Lmin+delta*(l+1)
            # print(lower_limit,upper_limit)

            # Create mask for each color bins
            y_hat = ((y_pred>=lower_limit)*1)*((y_pred<upper_limit)*1)
            y     = ((y_true>=lower_limit)*1)*((y_true<upper_limit)*1)

            # if there is no ground truth in this bin, dont include in calculation
            if torch.sum(y)==0:
                history.append(torch.tensor(0))
                continue
            # Compute dice score for current color bin
            temp = dice_score(y_hat,y,0)
            score.append(temp)
            history.append(temp)

        # print(score)
        return torch.mean(torch.stack(score), dim=0),history,delta
    
    
    elif isinstance(y_pred, np.ndarray):
        # create color bins and compute dice scores for each bin
        for l in range(L):
            lower_limit = Lmin+delta*l
            upper_limit = Lmin+delta*(l+1)
            # print(lower_limit,upper_limit)

            # Create mask for each color bins
            y_hat = ((y_pred>=lower_limit)*1)*((y_pred<upper_limit)*1)
            y     = ((y_true>=lower_limit)*1)*((y_true<upper_limit)*1)

            # if there is no ground truth in this bin, dont include in calculation
            if np.sum(y)==0:
                history.append(0)
                continue
            # Compute dice score for current color bin
            temp = dice_score(y_hat,y,0)
            score.append(temp)
            history.append(temp)
        return np.mean(score), history,delta
    return np.mean(score)
            
    # return 0


def compute_mse(y_pred,y_true):
    assert type(y_pred)==type(y_true)
    if isinstance(y_pred, torch.Tensor):
        score = torch.mean((y_pred -y_true)**2)
    elif isinstance(y_pred, np.ndarray):
        score = (np.square(y_pred -y_true)).mean()
    return score


def compute_masked_rc(y_pred,y_true, mask):
    '''
    compute pearson correlation coefficient
    '''
    score =  0
    assert type(y_pred)==type(y_true)
    total_elements = np.sum(mask)
    y_pred = y_pred*mask
    y_true = y_true*mask
    if total_elements ==0:
        return score
    else:
        # Calculate mean of x and y
        mean_p = np.sum(y_pred)/total_elements
        mean_t = np.sum(y_true)/total_elements

        # Calculate differences from mean
        diff_p = y_pred - mean_p
        diff_t = y_true - mean_t
        
        # Calculate numerator and denominator
        numerator = np.sum(diff_p * diff_t*mask)
        denominator = np.sqrt(np.sum((diff_p ** 2)*mask) * np.sum((diff_t ** 2)*mask))
        
        
        # Calculate Pearson correlation coefficient
        if denominator == 0:
            score =  0  # If denominator is 0, return 0 to avoid division by zero
        else:
            score = numerator / denominator
    return score

def compute_rc(y_pred,y_true):
    '''
    compute pearson correlation coefficient
    '''
    score =  0
    assert type(y_pred)==type(y_true)

    # Calculate mean of x and y
    mean_p = np.mean(y_pred)
    mean_t = np.mean(y_true)

    # Calculate differences from mean
    diff_p = y_pred - mean_p
    diff_t = y_true - mean_t
    
    # Calculate numerator and denominator
    numerator = np.sum(diff_p * diff_t)
    denominator = np.sqrt(np.sum(diff_p ** 2) * np.sum(diff_t ** 2))
    
    
    # Calculate Pearson correlation coefficient
    if denominator == 0:
        score =  0  # If denominator is 0, return 0 to avoid division by zero
    else:
        score = numerator / denominator
    return score


def compute_rel_rmse(y_pred,y_true,eps=0.0001):
    #formula rrmse = sqrt(avg[(1-pred/true)^2])
    y_pred=y_pred+eps
    y_true=y_true+eps
    assert type(y_pred)==type(y_true)
    if isinstance(y_pred, torch.Tensor):
        score = 100*torch.sqrt(torch.mean((1 -y_pred/y_true)**2))

    elif isinstance(y_pred, np.ndarray):
        score = 100*np.sqrt((np.square(1-y_pred/y_true)).mean())    
    return score


def compute_IoU(y_true, y_pred):
    assert type(y_pred)==type(y_true)

    if isinstance(y_pred, torch.Tensor):
        intersection = torch.logical_and(y_true, y_pred).sum()
        union = torch.logical_or(y_true, y_pred).sum()
        iou = intersection / union
        iou = iou.item()
    elif isinstance(y_pred, np.ndarray): 
        intersection = np.logical_and(y_true, y_pred).sum()
        union        = np.logical_or(y_true, y_pred).sum()
        iou = intersection / union      
    return iou

def dice_coefficient(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-7)  # Add a small epsilon to avoid division by zero
    return dice.item()

def pixel_accuracy(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        correct_pixels = torch.eq(y_true, y_pred).sum()
        total_pixels = y_true.numel()
        accuracy = correct_pixels / total_pixels
        accuracy = accuracy.item()
    elif isinstance(y_pred, np.ndarray): 
        correct_pixels = np.sum(np.equal(y_true, y_pred))
        total_pixels = np.size(y_true)
        accuracy = correct_pixels / total_pixels
        # accuracy = accuracy.item()        
    return accuracy

def mean_class_iou(y_true, y_pred, num_classes):
    iou_scores = []
    for class_idx in range(num_classes):
        class_true = y_true[class_idx,:,:]
        class_pred = y_pred[class_idx,:,:]
        iou = compute_IoU(class_true, class_pred)
        # print(iou)
        iou_scores.append(iou)
    mean_iou = sum(iou_scores) / len(iou_scores)
    return mean_iou



if __name__=="__main__":
    x = np.random.rand(1,3,3,2)
    y = np.random.rand(1,3,3,2)
    # x = np.array([10,20,30,40])
    # y = np.array([11,21,31,40])
    # mask = (x<21)*1
    # print(rmse(x,y))
    # print(masked_rmse(x,y,mask))
    # # print(rrmse(x,y))
    # print(compute_mse(x,y))

    x = (torch.from_numpy(x)).to(torch.float)
    y = (torch.from_numpy(y)).to(torch.float)
    # print(rmse(x,y))
    # print(compute_mse(x,y))
    

    # mask = (x<21)*1
    # # print(mask)
    # print(masked_rmse(x,y,mask))
    # # print(rrmse(x,y))
    x = np.array([[10,10],[10,0]])
    y = np.array([[5,8],[7,0.5]])
    # print(compute_rel_rmse(y,x,1))
    # # y = np.array([[0,0,1,1],[0,0,0,0]])
    # print(" Dice Score: ",dice_score(x,y))

    # x = (torch.from_numpy(x)).to(torch.float)
    # y = (torch.from_numpy(y)).to(torch.float)    
    # print(" Dice Score: ",dice_score(x,y))

    # print(compute_dice_score(x,y,5,0,5))

    def one_hot_encode(original_mask,classes):
    # print(classes)
        num_classes = len(classes)
        # print(num_classes)
        # Initialize an empty array for one-hot encoded masks
        encoded_masks = np.zeros((original_mask.shape[0], original_mask.shape[1], num_classes), dtype=int)

        # Encode the original mask as one-hot
        for id in range (num_classes):
            class_id = classes[id]
            encoded_masks[:, :, id] = (original_mask == class_id).astype(int)

        # The encoded_masks array will contain one-hot vectors for each pixel
        return encoded_masks

    # Define the original segmentation mask
    original_mask = np.array([[0, 0, 0, 1, 1],
                            [0, 1, 1, 2, 2],
                            [1, 1, 2, 2, 2],
                            [1, 1, 2, 2, 2]])
    y_true = one_hot_encode(original_mask,[0,1,2])
    
    pred_mask = np.array([[0, 0, 0, 1, 1],
                            [0, 1, 2, 2, 2],
                            [1, 1, 2, 2, 2],
                            [1, 1, 2, 2, 2]])
    y_pred = one_hot_encode(pred_mask,[0,1,2])
    
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    print(y_true.shape)

    print(IoU(y_true,y_pred))
    print(mean_class_iou(y_true,y_pred,3))