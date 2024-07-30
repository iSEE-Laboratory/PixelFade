import torch
import numpy as np

class PixelFade(object):
    def __init__(self, opt, logger, data_shape):
        self.opt = opt
        self.logger = logger    # if you want to print anything on log, feel free to use it by self.logger("hello world")
        self.replace_mask_list = self.prepare_replacement_mask_list(self.opt.fade_num_iter, data_shape)

    def prepare_replacement_mask_list(self, num_iter, data_shape):
        if num_iter==0:
            return None
        b, c, h, w = self.opt.batch_size, 3, data_shape[0], data_shape[1]
        data_len = b*c*h*w
        fade_rate_list = [1/num_iter for _ in range(num_iter)]
        fade_mask_list = []
        fade_mask = torch.ones((b, c, h, w)).view(-1).cuda()
        for i in range(num_iter):
            pick_indices = torch.where(fade_mask == 1)[0]
            pick_len = int(data_len*fade_rate_list[i])
            drop_indices = pick_indices[torch.randperm(len(pick_indices))[:pick_len]]
            fade_mask[drop_indices] = 0
            fade_mask_ = fade_mask.detach().clone()
            fade_mask_list.append(fade_mask_.bool())
        
        for i in range(num_iter-1, -1, -1):
            last_i = i-1
            if last_i != -1:
                fade_mask_list[i] = torch.logical_xor(fade_mask_list[i], fade_mask_list[last_i]).int()
            else:
                one_mask = torch.ones((b, c, h, w)).view(-1).cuda().bool()
                fade_mask_list[i] = torch.logical_xor(fade_mask_list[i], one_mask).int()
            
        return fade_mask_list


    def partial_pixel_replacement(self, step, prot_img, mask_list, accumulate_grad):
        mask = mask_list[step].cuda()
        b, c, h, w = prot_img.shape
        prot_img = prot_img.view(-1)
        if b!=self.opt.batch_size:
            mask = mask[:b*c*h*w]
            prot_img = prot_img[:b*c*h*w]
        
        prot_img[mask==1] = self.noise.view(-1)[mask==1]
        prot_img = prot_img.reshape((b, c, h, w))
        return prot_img

    def protect_image(self, model, data):
        # initialization
        prot_img = data.detach().clone()
        self.noise = torch.randn_like(data)
        max_num, min_num = torch.max(data), torch.min(data)
                
        sum_grad = torch.zeros_like(data)
        feature = model(data)
        
        condition = True   # True when loss<epsilon
        for step in range(self.opt.max_iter):
            #-----------Replacement Operation------------
            if step<self.opt.warmup_iter or condition==True: # warm up stage
                prot_img = self.partial_pixel_replacement(step=step%self.opt.fade_num_iter, 
                                                            prot_img=prot_img, 
                                                            mask_list=self.replace_mask_list, 
                                                            accumulate_grad=sum_grad)
                
            #-----------Constraint Operation------------
            prot_img = prot_img.detach().clone().cuda().requires_grad_(True)
            prot_feature = model(prot_img)
            
            model.zero_grad()

            loss_f_i = torch.mean((prot_feature - feature) ** 2)
            loss = loss_f_i 
            loss.backward(retain_graph=True)
            grad = prot_img.grad.data.clone()

            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            sum_grad = self.opt.momentum_w * sum_grad + grad
            prot_img.grad.data.zero_()
            prot_img = prot_img.data.clone()


            if loss_f_i<self.opt.epsilon and step<self.opt.max_iter-5:
                condition = True
                prot_img = prot_img - sum_grad * self.opt.beta
            else:
                # a small trick: set higher beta to make it go a step further, making it converge faster
                # beta_temp is 0.04 for market1501, 0.02 for cuhk
                condition = False
                prot_img = prot_img - sum_grad * self.opt.beta_temp

            prot_img = torch.clamp(prot_img, min=min_num, max=max_num)

        return prot_img.cpu().detach(), prot_feature.cpu().detach()
