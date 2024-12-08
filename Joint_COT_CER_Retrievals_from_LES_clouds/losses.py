'''
    Author: Zahid Hassan Tushar
    Email:  ztushar1@umbc.edu
'''

# Import libraries
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F
torch.manual_seed(1)

class segmentation_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self,y_true, y_pred):
        """
        Segmentation loss function using categorical cross-entropy.

        Args:
        - y_true: Ground truth segmentation masks (tensor of shape [batch_size, num_classes, height, width]).
        - y_pred: Predicted segmentation masks (tensor of the same shape as y_true).

        Returns:
        - Loss value (a scalar tensor).
        """

        # Ensure both y_true and y_pred have the same shape
        assert y_true.size() == y_pred.size(), "Input shapes do not match."

        # Calculate the categorical cross-entropy loss
        loss = nn.functional.cross_entropy(y_pred, y_true, reduction='mean')

        return loss


# define custom functions here
def find_mask (x,threshold1 = 0.5,threshold2=-0.5):

    # Create two masks for each channel based on the threshold values
    d = x.dim()
    if d==3:
        mask1 = x[0] >= threshold1
        mask2 = x[1] >= threshold2
        mask = torch.stack([mask1, mask2])
    elif d==4:
        mask1 = x[:, 0,] > threshold1
        mask2 = x[:, 1,] > threshold2

        mask = torch.stack(([mask1, mask2]), dim=1)
    else:
        raise ValueError ("Array should be 3D (C,H,W) or 4D (B,C,H,W)") 
    return mask

def find_inv_mask (x,threshold1 = 0.5,threshold2=-0.5):

    # Create two masks for each channel based on the threshold values
    d = x.dim()
    if d==3:
        mask1 = x[0] < threshold1
        mask2 = x[1] < threshold2
        mask = torch.stack([mask1, mask2])
    elif d==4:
        mask1 = x[:, 0,] < threshold1
        mask2 = x[:, 1,] < threshold2

        mask = torch.stack(([mask1, mask2]), dim=1)
    else:
        raise ValueError ("Array should be 3D (C,H,W) or 4D (B,C,H,W)") 
    return mask

class WeightedMSE(torch.nn.Module):
    def __init__(self,alpha,threshold):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.alpha = alpha
        self.threshold = threshold

        
    def forward(self, pred, actual):
        mask = (actual>=self.threshold)*1
        inv_mask = (actual<self.threshold)*1
        a=self.mse(pred*inv_mask,actual*inv_mask)
        b=self.mse(pred*mask,actual*mask)*self.alpha

        return a+b

class WeightedMSE2(torch.nn.Module):
    def __init__(self,alpha,th1,th2):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.alpha = alpha
        self.th1 = th1
        self.th2 = th2

        
    def forward(self, pred, actual):
        mask = find_mask(actual,self.th1,self.th2)*1
        inv_mask = find_inv_mask(actual,self.th1,self.th2)*1
        a=self.mse(pred*inv_mask,actual*inv_mask)
        b=self.mse(pred*mask,actual*mask)*self.alpha

        return a+b


class MSE_KL_Loss(torch.nn.Module):
    def __init__(self,beta):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.kld = KLD_Loss()
        self.beta =beta
        
    def forward(self, pred, actual):
        return self.mse(pred,actual)+self.beta*self.kld(pred,actual)

class KLD_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    def get_probability(self,x):
        x = x.cpu()
        lm1 = torch.log(torch.tensor(0.01)).item()
        lm2 = torch.log(torch.tensor(400)).item()
        x,_ = torch.histogram(x,100,range=(lm1,lm2))
        x = x+0.00001
        x = x/sum(x)
        return x

    def forward(self, pred, actual):
        pred = self.get_probability(pred)
        actual = self.get_probability(actual)
        return self.kld(torch.log(pred.detach()),torch.log(actual.detach()))


class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return (self.mse(torch.log(pred + 1-torch.min(pred)), torch.log(actual + 1-torch.min(actual))))

class KL_Loss_W_PENALTY(torch.nn.Module):
    def __init__(self,alpha,beta,threshold):
        super().__init__()
        self.weighted_mse = WeightedMSE(alpha,threshold)
        self.kld = KLD_Loss2()
        self.beta =beta
        
    def forward(self, pred, actual):
        return (self.weighted_mse(pred,actual)+self.beta*self.kld(pred,actual))/2

class SSIM_W_PENALTY(torch.nn.Module):
    def __init__(self,alpha,threshold):
        super().__init__()
        self.weighted_mse = WeightedMSE(alpha,threshold)
        
    def forward(self, pred, actual):
        a = np.squeeze(pred.clone().cpu().detach().numpy())
        b = np.squeeze(actual.clone().cpu().detach().numpy())
        # print(a.shape)
        # print(b.shape)
        return self.weighted_mse(pred,actual)-ssim(a,b)+1

class SSIM_W_PENALTY2(torch.nn.Module):
    def __init__(self,alpha,threshold):
        super().__init__()
        self.weighted_mse = WeightedMSE(alpha,threshold)
        
    def forward(self, pred, actual):
        a = np.squeeze(pred.clone().cpu().detach().numpy())
        b = np.squeeze(actual.clone().cpu().detach().numpy())
        return self.weighted_mse(pred,actual)*(-ssim(a,b)+1)

class SSIM_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        a = np.squeeze(pred.clone().cpu().detach().numpy())
        b = np.squeeze(actual.clone().cpu().detach().numpy())
        return ssim(a,b)

class SSIM_W_PENALTY3(torch.nn.Module):
    def __init__(self,alpha,threshold):
        super().__init__()
        self.weighted_mse = WeightedMSE(alpha,threshold)
        self.ssim = SSIM()
        
    def forward(self, pred, actual):
        a = pred.clone().cpu().detach().numpy()
        b = actual.clone().cpu().detach().numpy()
        return self.weighted_mse(pred,actual)-self.ssim(a,b)+1

class SSIM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        if isinstance(pred, torch.Tensor):
            # print("Oh Yeah")
            a = pred.clone().cpu().detach().numpy()
            b = actual.clone().cpu().detach().numpy()
            loss = []
            assert a.shape==b.shape
            if a.ndim==3:
                for i in range (a.shape[0]):
                    loss.append(ssim(a[i],b[i]))
            elif a.ndim==4:
                for i in range (a.shape[0]):
                    for j in range (a.shape[1]):
                        loss.append(ssim(a[i,j],b[i,j]))

        elif isinstance(pred, np.ndarray):
            # print("Oh No!")
            a = pred
            b = actual
            loss = []
            assert a.shape==b.shape
            if a.ndim==3:
                for i in range (a.shape[0]):
                    loss.append(ssim(a[i],b[i]))
            elif a.ndim==4:
                for i in range (a.shape[0]):
                    for j in range (a.shape[1]):
                        loss.append(ssim(a[i,j],b[i,j]))
                   
        return np.average(loss)

class SSIM_W_PENALTY4(torch.nn.Module):
    def __init__(self,alpha,threshold):
        super().__init__()
        self.weighted_mse = WeightedMSE(alpha,threshold)
        self.ssim = SSIM4()
        
    def forward(self, pred, actual):
        a = pred.clone().cpu().detach().numpy()
        b = actual.clone().cpu().detach().numpy()
        return self.weighted_mse(pred,actual)-self.ssim(a,b)+1

class SSIM4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        if isinstance(pred, torch.Tensor):
            # print("Oh Yeah")
            a = pred.clone().cpu().detach().numpy()
            b = actual.clone().cpu().detach().numpy()
            loss = []
            assert a.shape==b.shape
            if a.ndim==3:
                for i in range (a.shape[0]):
                    loss.append(ssim(a[i],b[i]))
            elif a.ndim==4:
                for i in range (a.shape[0]):
                    for j in range (a.shape[1]):
                        loss.append(ssim(a[i,j],b[i,j],win_size=5))

        elif isinstance(pred, np.ndarray):
            # print("Oh No!")
            a = pred
            b = actual
            loss = []
            assert a.shape==b.shape
            if a.ndim==3:
                for i in range (a.shape[0]):
                    loss.append(ssim(a[i],b[i]))
            elif a.ndim==4:
                for i in range (a.shape[0]):
                    for j in range (a.shape[1]):
                        loss.append(ssim(a[i,j],b[i,j],win_size=5))
                   
        return np.average(loss)

class KLD_Loss2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction="batchmean", log_target=True)
    
    def get_probability(self,x):
        if isinstance(x, torch.Tensor):
            x = F.log_softmax(x.flatten(),dim=0) 

        elif isinstance(x, np.ndarray):
            x = torch.tensor(x) 
            x = F.log_softmax(x.flatten(),dim=0)  

        return x

        
    def forward(self, pred, actual):
        pred = self.get_probability(pred)
        actual = self.get_probability(actual)
        return self.kld(pred,actual)


class Joint_L2_L1(torch.nn.Module):
    '''
    computes MSE for CER and COT separately and returns the losses together as follow
    loss = (w1 * COT_loss+ w2 *CER_loss)/2
    '''
    def __init__(self,w1, w2):
        super().__init__()
        self.w1 = w1 
        self.w2 = w2
        self.mse = torch.nn.MSELoss()   
        self.l1loss = torch.nn.L1Loss()

    def forward(self, pred, actual):
        COT_loss = 0
        CER_loss = 0
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape

        if pred.ndim==3:
            COT_loss = self.mse(pred[0,:,:],actual[0,:,:])
            CER_loss = self.l1loss(pred[1,:,:],actual[1,:,:])
        elif pred.ndim==4:
            COT_loss = self.mse(pred[:,0,:,:],actual[:,0,:,:])
            CER_loss = self.l1loss(pred[:,1,:,:],actual[:,1,:,:])            
        return (self.w1*COT_loss+self.w2*CER_loss)/2


class Joint_Penalty_L2(torch.nn.Module):
    '''
    computes MSE for CER and COT separately and returns the losses together as follow
    loss = (w1 * COT_loss+ w2 *CER_loss)/2
    '''
    def __init__(self,w1, w2,alpha,threshold):
        super().__init__()
        self.w1 = w1 
        self.w2 = w2
        self.mse = torch.nn.MSELoss()   
        self.penalty = SSIM_W_PENALTY3(alpha,threshold)  

    def forward(self, pred, actual):
        COT_loss = 0
        CER_loss = 0
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape

        if pred.ndim==3:
            COT_loss = self.penalty(pred[0,:,:],actual[0,:,:])
            CER_loss = self.mse(pred[1,:,:],actual[1,:,:])
        elif pred.ndim==4:
            COT_loss = self.penalty(pred[:,0,:,:],actual[:,0,:,:])
            CER_loss = self.mse(pred[:,1,:,:],actual[:,1,:,:])            
        return (self.w1*COT_loss+self.w2*CER_loss)/2

class Joint_Penalty_L2_6x(torch.nn.Module):
    '''
    FOR model that predicts 6x6
    computes MSE for CER and Penalty for COT separately and returns the losses together as follow
    loss = (w1 * COT_loss+ w2 *CER_loss)/2
    '''
    def __init__(self,w1, w2,alpha,threshold):
        super().__init__()
        self.w1 = w1 
        self.w2 = w2
        self.mse = torch.nn.MSELoss()   
        self.penalty = SSIM_W_PENALTY4(alpha,threshold)  

    def forward(self, pred, actual):
        COT_loss = 0
        CER_loss = 0
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape

        if pred.ndim==3:
            COT_loss = self.penalty(pred[0,:,:],actual[0,:,:])
            CER_loss = self.mse(pred[1,:,:],actual[1,:,:])
        elif pred.ndim==4:
            COT_loss = self.penalty(pred[:,0,:,:],actual[:,0,:,:])
            CER_loss = self.mse(pred[:,1,:,:],actual[:,1,:,:])            
        return (self.w1*COT_loss+self.w2*CER_loss)/2



class Joint_L2(torch.nn.Module):
    '''
    computes MSE for CER and COT separately and returns the losses together as follow
    loss = (w1 * COT_loss+ w2 *CER_loss)/2
    '''
    def __init__(self,w1, w2,):
        super().__init__()
        self.w1 = w1 
        self.w2 = w2
        self.mse = torch.nn.MSELoss()     

    def forward(self, pred, actual):
        COT_loss = 0
        CER_loss = 0
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape


        if pred.ndim==3:
            COT_loss = self.mse(pred[0,:,:],actual[0,:,:])
            CER_loss = self.mse(pred[1,:,:],actual[1,:,:])
        elif pred.ndim==4:
            COT_loss = self.mse(pred[:,0,:,:],actual[:,0,:,:])
            CER_loss = self.mse(pred[:,1,:,:],actual[:,1,:,:])            
        return (self.w1*COT_loss+self.w2*CER_loss)/2


class Joint_L2_Binary(torch.nn.Module):
    '''
    computes MSE for CER and COT, and BinaryCrossEntropy for Cloud Mask separately,and returns the losses together as follow
    loss = (w1 * COT_loss + w2 *CER_loss + w3 * Mask_Loss)/2
    '''
    def __init__(self,w1, w2,w3):
        super().__init__()
        self.w1 = w1 
        self.w2 = w2
        self.w3 = w3
        self.mse = torch.nn.MSELoss()   
        self.m_loss = nn.BCELoss()

    def forward(self, pred, actual):
        COT_loss = 0
        CER_loss = 0
        Mask_loss = 0
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape


        if pred.ndim==3:
            COT_loss  = self.mse(pred[0,:,:],actual[0,:,:])
            CER_loss  = self.mse(pred[1,:,:],actual[1,:,:])
            Mask_loss = self.m_loss(pred[2,:,:],actual[2,:,:])
        elif pred.ndim==4:
            COT_loss  = self.mse(pred[:,0,:,:],actual[:,0,:,:])
            CER_loss  = self.mse(pred[:,1,:,:],actual[:,1,:,:])  
            Mask_loss = self.m_loss(pred[:,2,:,:],actual[:,2,:,:])          
        return (self.w1*COT_loss+self.w2*CER_loss+self.w3*Mask_loss)/3


def apply_mask(data,mask):
    binary_mask = (mask>0)*1
    ret = data*binary_mask
    ret[ret==0]=np.log(0.01)
    return ret

class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, predicted, target):
        # Calculate Euclidean distance
        euclidean_distance = torch.sqrt(torch.sum((predicted - target)**2, dim=1))
        
        # You can choose to return the mean of the distances or the sum
        loss = torch.mean(euclidean_distance)
        return loss


class IoU(torch.nn.Module):
    '''
    computes MSE for CER and COT, and BinaryCrossEntropy for Cloud Mask separately,and returns the losses together as follow
    loss = (w1 * COT_loss + w2 *CER_loss + w3 * Mask_Loss)/2
    '''
    def __init__(self,eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def iou(self,pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum()
        union = torch.logical_or(pred_mask, true_mask).sum()
        
        iou_score = (intersection.float()+self.eps )/ (union.float()+self.eps )
        return iou_score
    
    def forward(self, pred, actual):
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape

        if pred.ndim==2:
            loss = self.iou(pred,actual)
        elif pred.ndim==3:
            loss =self.iou(pred[0,:,:],actual[0,:,:])
        elif pred.ndim==4:
            loss = self.iou(pred[0,0,:,:],actual[0,0,:,:])
     
        return loss

def iou(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    
    iou_score = intersection.float() / union.float()
    return iou_score

def is_binary_tensor(tensor):
    unique_values = torch.unique(tensor)
    return torch.all(unique_values == 0) or torch.all(unique_values == 1)

class RegularizedLoss(nn.Module):
    def __init__(self,lamda):
        super(RegularizedLoss, self).__init__()
        self.lamda = lamda
        self.mse = nn.MSELoss()

    def forward(self, predicted, actual):
        assert type(pred)==type(actual)
        assert pred.shape==actual.shape
        mse_loss = self.mse(predicted, actual) 
        if pred.ndim==3:
            low_pred_mask = predicted[1,:,:] < 0.1
            penalized_loss = torch.mean((0.1 - predicted[1,:,:])[low_pred_mask]**2) * self.lamda
        elif pred.ndim==4:
            low_pred_mask = predicted[:,1,:,:] < 0.1
            penalized_loss = torch.mean((0.1 - predicted[:,1,:,:])[low_pred_mask]**2) * self.lamda
        
        return mse_loss + penalized_loss


def soft_iou_loss(logits, labels):
    """
    Computes the soft IoU loss.
    Args:
        logits: model logits predictions, shape (N, C, H, W)
        labels: ground truth labels, shape (N, H, W) 
    """
    num_classes = logits.shape[1]

    logits = F.softmax(logits, dim=1)
    
    dice_numerator = 0
    dice_denominator = 0
    for cls in range(num_classes):
        probability = logits[:, cls, :, :] # (N, H, W)
        label = (labels == cls).float() # (N, H, W)
        inter = (probability * label).sum((1, 2))
        union = probability.sum((1, 2)) + label.sum((1, 2))
        dice_class = (2 * inter + 1) / (union + 1) # (N,)
        dice_numerator += dice_class
        dice_denominator += 1
    
    return 1 - dice_numerator/dice_denominator

class soft_DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,logits, labels):
        """
        Computes the soft Dice loss for binary segmentation. 
        Args:
            logits: predicted masks from model, shape (N, 1, H, W) 
            labels: ground truth masks, shape (N, H, W)
        """

        # logits = torch.sigmoid(logits) # sigmoid to map values between 0 and 1

        pred = logits.view(-1).float() # (N * H * W,)
        label = labels.view(-1).float() # (N * H * W,)

        inter = (pred * label).sum()
        union = pred.sum() + label.sum()
        
        dice_class = (2 * inter + 1) / (union + 1)
        
        return 1 - dice_class

class soft_IoULoss(nn.Module):
    def __init__(self):
        super().__init__() 
        
    def forward(self, logits, labels):
        # probas = torch.sigmoid(logits) 
        inter = torch.sum(logits * labels)
        union = torch.sum(logits) + torch.sum(labels) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return 1 - iou

class BCE_IoULoss(nn.Module):
    def __init__(self,w1):
        super().__init__() 
        self.weight =w1
        self.bce = torch.nn.BCELoss()
        
    def forward(self, logits, labels):
        # probas = torch.sigmoid(logits) 
        bce_loss = self.bce(logits, labels) 
        inter = torch.sum(logits * labels)
        union = torch.sum(logits) + torch.sum(labels) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return (1-self.weight)*(1 - iou)+self.weight*bce_loss


if __name__=="__main__":
    
    # Create dummy tensors 
    logits = torch.tensor([[[[0.3, 0.7, 0.9], 
                            [0.8, 0.2, 0.2]]]]) # (N=1, C=1, H=2, W=2)  

    labels = torch.tensor([[[[1, 0,1], 
                            [1, 0,0]]]]) # (N=1, H=2, W=2)

    # # Compute soft IoU loss
    # loss = soft_iou_binary_loss(logits, labels)
    # print(loss)

    criterion = soft_IoULoss()
    print(criterion(logits,labels))
    # cot = torch.randn((256,3,6,6)).float()

    # # pred = torch.randn((256,2,6,6)).float()
    # # pred = torch.randint(10,(10,10)).float()
    # pred = cot.clone()*0.9
    # cot[:,2,:,:] = 0
    # # pred.to(device='cuda')
    # # cot.to(device='cuda')
    # # print(cot)
    # # print(pred)
    # alpha = 5.0
    # beta=100
    # threshold=1.69
    # criterion = WeightedMSE(alpha,threshold)
    # loss =criterion(pred,cot)

    # th1=15
    # th2=9
    # criterion = WeightedMSE2(alpha,th1,th2)
    # loss =criterion(pred,cot)

    # criterion2 = SSIM_W_PENALTY3(alpha,threshold)
    # loss = criterion2(pred,cot)
    # print(loss)

    # criterion3 = SSIM_loss()
    # loss = criterion3(pred,cot)
    # print(loss)

    # criterion3 = SSIM4()
    # loss= criterion3(pred,cot)
    # print(cot.min())
    # print(loss)
    # criterion = SSIM4()

    # criterion = Joint_L2_Binary(1,10,1)
    # loss = criterion(pred,cot)
    # print(loss)


    
    # a = torch.tensor([0,2,3,4,5,6,7,8,9,0]).float()
    # b = torch.tensor([1,2,3,4,5,6,7,8,9,10]).float()
    # critereion = RegularizedLoss(10)
    # loss = critereion(pred,cot)
    # print(loss)

    # c1 = torch.nn.MSELoss()
    # loss1 = c1(a,b)
    # print("MSE:  ",loss1)

    # c2 = KLD_Loss2()
    # loss2 = c2(a,b)
    # print("KL:  ",torch.exp(loss2) ) 
    # 
    # # Example usage
    # # Assuming pred_mask and true_mask are binary masks of the same shape
    # pred_mask = torch.tensor([[1, 0, 1],
    #                         [0, 1, 0],
    #                         [1, 0, 0]], dtype=torch.bool)

    # true_mask = torch.tensor([[1, 0, 0],
    #                         [0, 1, 1],
    #                         [1, 1, 0]], dtype=torch.bool)
    
    # pred_mask = torch.randn((10,1,6,6)).float()

    # iou_score = iou(pred_mask, true_mask)
    # print("IoU Score:", iou_score.item())  

    # criterion = IoU()
    # iou_score = criterion(pred_mask, true_mask)
    # print("Class IoU Score:", iou_score.item())   