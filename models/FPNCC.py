import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        

        if model_name == 'Res50_FPN':
            from FPNCC_Model.Res50_FPN import Res50_FPN as net
        elif model_name == 'Res101_FPN':
            from FPNCC_Model.Res101_FPN import Res101_FPN as net

        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse, self.loss_mse_p2, self.loss_mse_p3, self.loss_mse_p4
    
    def forward(self, img, gt_map):                               
        density_map, density_map_p2, density_map_p3, density_map_p4 = self.CCN(img)                          
        self.loss_mse, self.loss_mse_p2, self.loss_mse_p3, self.loss_mse_p4 = self.build_loss(
            density_map.squeeze(), 
            density_map_p2.squeeze(), 
            density_map_p3.squeeze(), 
            density_map_p4.squeeze(), 
            gt_map.squeeze())               
        return density_map
    
    def build_loss(self, density_map, density_map_p2, density_map_p3, density_map_p4, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data) 
        loss_mse_p2 = self.loss_mse_fn(density_map_p2, gt_data)   
        loss_mse_p3 = self.loss_mse_fn(density_map_p3, gt_data)  
        loss_mse_p4 = self.loss_mse_fn(density_map_p4, gt_data)  
        return loss_mse, loss_mse_p2, loss_mse_p3, loss_mse_p4

    def test_forward(self, img):                               
        density_map, _, _, _ = self.CCN(img)                    
        return density_map

