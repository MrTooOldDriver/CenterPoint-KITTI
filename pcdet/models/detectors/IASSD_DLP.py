from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn

class IASSD_DLP(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        print('building IA-SSD')

        # halting stuff
        MAX_STEPS = 3
        LAMBDA_P = 0.3
        self.BETA = 0.1
        p_g = torch.zeros((MAX_STEPS,))
        not_halted = 1.
        for k in range(MAX_STEPS):
            p_g[k] = not_halted * LAMBDA_P
            not_halted = not_halted * (1 - LAMBDA_P)
        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # process for output
            batch_dict = self.get_halted_output(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_halted_output(self, batch_dict):
        halting_step = batch_dict['halting_step'].cpu().numpy()
        batch_dict['batch_index'] = batch_dict['0_batch_index'].clone()
        batch_dict['batch_box_preds'] = batch_dict['0_batch_box_preds'].clone()
        batch_dict['batch_cls_preds'] = batch_dict['0_batch_cls_preds'].clone()
        batch_dict['cls_preds_normalized'] = batch_dict['0_cls_preds_normalized']
        for i, halt_layer in enumerate(halting_step):
            batch_dict['batch_index'][i] = batch_dict['%s_batch_index' % (halt_layer-1)][i]
            batch_dict['batch_box_preds'][i] = batch_dict['%s_batch_box_preds' % (halt_layer-1)][i]
            batch_dict['batch_cls_preds'][i] = batch_dict['%s_batch_cls_preds' % (halt_layer-1)][i]
        return batch_dict

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_point_list, tb_dict_list = self.point_head.get_loss()
        loss = 0

        for ponder_layer, tb_dict in tb_dict_list.items():
            for key in tb_dict.keys():
                if 'loss' in key:
                    # if 'sa' in key:
                    #     pass
                    # else:
                    #     disp_dict[str(ponder_layer)+key] = tb_dict[key]
                    disp_dict[str(ponder_layer)+'_det_loss'] = tb_dict['det_loss']
        

        # calculate pendor loss
        # reconstruction loss
        p = batch_dict['halting_p'] 
        reconstruction_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * loss_point_list[n]).mean()
            reconstruction_loss = reconstruction_loss + loss
        disp_dict['recon_loss'] = reconstruction_loss.item()
        loss += reconstruction_loss

        # regularization loss
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        reg_loss = self.kl_div(p.log(), p_g)
        reg_loss = reg_loss * self.BETA
        disp_dict['reg_loss'] = reg_loss.item()
        loss += reg_loss

        return loss, tb_dict, disp_dict