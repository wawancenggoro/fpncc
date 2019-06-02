import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.FPNCC import CrowdCounter
from config import cfg
from misc.utils import *
import pdb


class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()
        print(self.net)
        print('Use model: {}'.format(cfg.NET))
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 
        # self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp')


        self.i_tb = 0
        self.epoch = -1

        #resume
        self.start_epoch = 0
        self.net.load_state_dict(torch.load("../models/we.pth"))

        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.restore_transform = dataloader()


    def forward(self):

        # validation
        self.timer['val time'].tic()
        if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
            self.validate_V1()
        elif self.data_mode is 'WE':
            self.validate_V2()
        elif self.data_mode is 'GCC':
            self.validate_V3()
        self.timer['val time'].toc(average=False)
        print 'val time: {:.2f}s'.format(self.timer['val time'].diff)

        # for epoch in range(cfg.MAX_EPOCH-self.start_epoch):
        #     self.epoch = epoch + self.start_epoch
        #     if epoch > cfg.LR_DECAY_START:
        #         self.scheduler.step()
                
        #     # training    
        #     self.timer['train time'].tic()
        #     self.train()
        #     self.timer['train time'].toc(average=False)

        #     print 'train time: {:.2f}s'.format(self.timer['train time'].diff)
        #     print '='*20

        #     # validation
        #     if epoch%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
        #         self.timer['val time'].tic()
        #         if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50']:
        #             self.validate_V1()
        #         elif self.data_mode is 'WE':
        #             self.validate_V2()
        #         elif self.data_mode is 'GCC':
        #             self.validate_V3()
        #         self.timer['val time'].toc(average=False)
        #         print 'val time: {:.2f}s'.format(self.timer['val time'].diff)


    def train(self): # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)
            loss_all, loss_p2, loss_p3, loss_p4 = self.net.loss
            loss = loss_all+loss_p2+loss_p3+loss_p4
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff)
                print '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg_data.LOG_PARA, pred_map[0].sum().data/self.cfg_data.LOG_PARA)            


    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        maes_ls = []
        mses_ls = []
        gt_count_ls = []

        for vi, data in enumerate(self.val_loader, 0):
            import pdb; pdb.set_trace()
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    
                    loss_all, loss_p2, loss_p3, loss_p4 = self.net.loss
                    loss = loss_all.item()+loss_p2.item()+loss_p3.item()+loss_p4.item()
                    losses.update(loss)
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))

                    maes_ls.append(abs(gt_count-pred_cnt))
                    gt_count_ls.append(gt_count)
                    # mses_ls.append(np.sqrt((gt_count-pred_cnt)*(gt_count-pred_cnt)))

                    print '        [cnt: gt: %.1f pred: %.2f]' % (gt_count, pred_cnt) 

                # if vi==0:
                #     vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        # self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        # self.writer.add_scalar('mae', mae, self.epoch + 1)
        # self.writer.add_scalar('mse', mse, self.epoch + 1)

        # self.train_record = update_model(self.net,self.epoch,self.exp_path,self.exp_name,[mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)

        from matplotlib import pyplot as plt
        
        import pandas as pd
        df = pd.DataFrame(columns=['mae','gt_count'])
        df['mae'] = maes_ls
        df['gt_count'] = gt_count_ls  
        df_sort = df.sort_values(by=['gt_count'])
        import math
        plt.close()
        plt.plot(df_sort['gt_count'], df_sort['mae'], label='Absolute Error')
        maes_max = int(math.ceil(np.array(maes_ls).max()))
        gt_count_max = int(math.ceil(np.array(df_sort['gt_count']).max()))
        plt.ylim(maes_max+5,0)
        mae_line = np.array([mae for i in xrange(gt_count_max)])
        mse_line = np.array([mse for i in xrange(gt_count_max)])
        plt.plot(mae_line, 'r--', label='MAE') 
        plt.plot(mse_line, 'g--', label='MSE') 
        plt.xlabel("Ground Truth Count")
        plt.ylabel("Absolute Error")
        plt.legend()
        plt.savefig('maes.jpg')

        # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        # ax1.plot(maes_ls)
        # maes_max = np.array(maes_ls).max().round().astype(int)
        # ax1.ylim(maes_max+1,0)
        # plt.savefig('maes.jpg')

        # plt.plot(mses_ls)
        # plt.savefig('mses.jpg')


    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        maes_ls = [],[],[],[],[]
        gt_count_ls = [],[],[],[],[]

        roi_mask = []
        from datasets.WE.setting import cfg_data 
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:

            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH,'test',val_folder + '_roi.mat'))['BW'])
        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img,gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                    
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        loss_all, loss_p2, loss_p3, loss_p4 = self.net.loss
                        loss = loss_all.item()+loss_p2.item()+loss_p3.item()+loss_p4.item()
                        losses.update(loss,i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)

                        maes_ls[i_sub].append(abs(gt_count-pred_cnt))
                        gt_count_ls[i_sub].append(gt_count)

                        print '        [cnt: gt: %.1f pred: %.2f]' % (gt_count, pred_cnt) 
                    # if vi==0:
                    #     vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        # self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        # self.writer.add_scalar('mae', mae, self.epoch + 1)
        # self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        # self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        # self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        # self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        # self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        # self.train_record = update_model(self.net,self.epoch,self.exp_path,self.exp_name,[mae, 0, loss],self.train_record,self.log_txt)
        # print_WE_summary(self.log_txt,self.epoch,[mae, 0, loss],self.train_record,maes)
        print(mae)

        from matplotlib import pyplot as plt
        import IPython; IPython.embed()
        
        import pandas as pd
        for i_sub in range(5):
            df = pd.DataFrame(columns=['mae','gt_count'])
            df['mae'] = maes_ls[i_sub]
            df['gt_count'] = gt_count_ls[i_sub]
            df_sort = df.sort_values(by=['gt_count'])
            import math
            plt.close()
            plt.plot(df_sort['gt_count'], df_sort['mae'], label='Absolute Error')
            maes_max = int(math.ceil(np.array(maes_ls[i_sub]).max()))
            gt_count_max = int(math.ceil(np.array(df_sort['gt_count']).max()))
            plt.ylim(maes_max+5,0)
            mae_line = np.array([maes.avg[i_sub] for i in xrange(gt_count_max)])
            plt.plot(mae_line, 'r--', label='MAE') 
            plt.xlabel("Ground Truth Count")
            plt.ylabel("Absolute Error")
            plt.legend()
            plt.savefig('we%d.jpg' % (i_sub+1))





    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)   
                    attributes_pt = attributes_pt.squeeze() 
                    c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    c_mses['weather'].update(s_mse,attributes_pt[i_img][2])


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.epoch,self.exp_path,self.exp_name,[mae, mse, loss],self.train_record)

        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record,c_maes,c_mses)
