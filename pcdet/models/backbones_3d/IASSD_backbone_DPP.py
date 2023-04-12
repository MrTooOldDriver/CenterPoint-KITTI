import pathlib
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
import os

class IASSD_Backbone_DPP(nn.Module): # Dynamic Point Ponder
    """ Backbone for IA-SSD"""

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class
        

        self.SA_modules = nn.ModuleList()
        self.SA_router_modules = nn.ModuleList()
        self.SA_upsample_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)
        self.dpp_layers = sa_config.DPP_LAYER

        # =====================================================
        # add options in backbone configuration, a path to save 
        # intermediate features during grouping at inference
        # self.save_features_dir = path/to/save or None
        # =====================================================
        self.save_features_dir = model_cfg.get('SAVE_FEAT_DIR', None)
        # only use pooling for visualizations this option is injected in the main script
        self.use_pooling_weight = model_cfg.get('USE_POOLING_WEIGHT', False)

        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list): ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,                                                
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class = self.num_class,
                        use_pooling_weights=self.use_pooling_weight,
                        sample_idx=True
                    )
                )

                if k in self.dpp_layers:
                    dpp_index = self.dpp_layers.index(k)
                    dpp_channel_in = sa_config.DPP_CHANNEL_IN[dpp_index]
                    dpp_channel_out = sa_config.DPP_CHANNEL_OUT[dpp_index]
                    self.SA_router_modules.append(
                        DynamicPointPonderRouterMLP(feat_num=dpp_channel_in)
                    )
                    self.SA_upsample_modules.append(
                        # DynamicPointPonderUpSampling(input_feat_num=dpp_channel_in, output_feat_num=dpp_channel_out)
                        DynamicPointPonderConvUpSampling(mlps=sa_config.MLPS[k], input_feat_num=dpp_channel_in, output_feat_num=dpp_channel_out)
                    )
                    print('Create DPP layer Dynamic Module at layer %s channel_in %s channel_out %s' % (k, channel_in, channel_out))
                else:
                    self.SA_router_modules.append(None)
                    self.SA_upsample_modules.append(None)

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range
                                                                    )
                                       )
            elif self.layer_types[k] == 'PCT_Layer':
                mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
                try:
                    for idx in range(mlps.__len__()):
                        mlps[idx] = [channel_in] + mlps[idx]
                except:
                    pass
                self.SA_modules.append(
                    pointnet2_modules.AttentiveSAModule(
                        npoint_list=self.model_cfg.SA_CONFIG.NPOINT_LIST[k],
                        radii=self.model_cfg.SA_CONFIG.RADIUS_LIST[k],
                        nsamples=self.model_cfg.SA_CONFIG.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,
                        out_channel=self.model_cfg.SA_CONFIG.AGGREGATION_MLPS[k][0]
                    )
                )
            
                channel_out = self.model_cfg.SA_CONFIG.AGGREGATION_MLPS[k][0]
            channel_out_list.append(channel_out)

        self.num_point_features = channel_out

        self.total_gates = len(self.dpp_layers)
        self.gate_static = [0 for _ in range(self.total_gates)]
        self.keep_gate_static = [0 for _ in range(self.total_gates)]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        # inital DPP gate
        current_routing_gate = torch.ones(size=xyz.shape[0:2]).cuda()
        dpp_gates = []
        dpp_inverted_gates = []
        dpp_upsampled_features = []
        dpp_original_features = []

        li_cls_pred = None
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                # =====================================================
                # input parameter during the SA_Layer forward to save intermediate points
                # =====================================================
                if self.training:
                    save_path = None
                elif self.save_features_dir is None:
                    save_path = None
                else:
                    from pathlib import Path
                    layer_name = 'Layer_' + str(i)
                    save_path = Path(self.save_features_dir) / layer_name
                    
                    save_path = str(save_path)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                # construct a saving path for different layer
                if i in self.dpp_layers:
                    # update gate using current feat
                    new_routing_gate = self.SA_router_modules[i](feature_input).squeeze()
                    # if self.training or len(new_routing_gate.shape) == 3:
                    #     new_routing_gate = F.gumbel_softmax(new_routing_gate, dim=-1, hard=True)
                    #     new_routing_gate = new_routing_gate[:, :, 0].contiguous()
                    # else: 
                    #     new_routing_gate = F.gumbel_softmax(new_routing_gate, dim=-1, hard=True)
                    #     new_routing_gate = new_routing_gate[:, 0].unsqueeze(0).contiguous()
                    #     #TODO Folloing code isn't right, fix it later plz
                    #     # new_routing_gate = new_routing_gate.max(-1, keepdim=True)[1]
                    #     # new_routing_gate = new_routing_gate.permute(1,0)
                    #     # new_routing_gate = new_routing_gate.type(torch.float32)
                    #     # new_routing_gate = torch.ones_like(new_routing_gate) - new_routing_gate
                    current_routing_gate = new_routing_gate
                    if len(current_routing_gate.shape) == 1:
                        current_routing_gate = current_routing_gate.unsqueeze(0)
                # TODO use gate masking to do the masking then do sampling
                dpp_gates.append(current_routing_gate)

                li_xyz, li_features, li_cls_pred, li_sampled_idx_list = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz, save_features_dir=save_path, frame_id=batch_dict['frame_id'][0])
                # print('index i=', i)
                # print('feature_input.shape:', feature_input.shape)
                # print('xyz_input.shape:', xyz_input.shape)
                # print('li_xyz.shape: ', li_xyz.shape)
                # print('li_features.shape: ', li_features.shape)
                # print('ctr_xyz.shape: ', ctr_xyz.shape) if ctr_xyz is not None else None
                # print('li_cls_pred.shape: ', li_cls_pred.shape) if li_cls_pred is not None else None

                # DPP
                if i in self.dpp_layers:
                    assert current_routing_gate.shape[1] == xyz_input.shape[1]
                    dpp_layer_idx = self.dpp_layers.index(i)
                    # find correspondence
                    # import ipdb; ipdb.set_trace()
                    # routing_gate_for_gather = current_routing_gate.unsqueeze(-1).permute(0,2,1)
                    routing_gate_for_gather = current_routing_gate.unsqueeze(-1).permute(0,2,1).contiguous()
                    correspondence_gate = pointnet2_utils.gather_operation(routing_gate_for_gather, li_sampled_idx_list).contiguous()
                    correspondence_original_feat = pointnet2_utils.gather_operation(feature_input, li_sampled_idx_list).contiguous()
                    correspondence_invert_gate = (torch.ones_like(correspondence_gate) - correspondence_gate).contiguous()
                    correspondence_original_feat_upsample = self.SA_upsample_modules[dpp_layer_idx](correspondence_original_feat)
                    # correspondence_original_feat_upsample = F.pad(input, li_features.shape(), mode='constant', value=None)
                    
                    li_features_for_dpp = li_features.clone().detach() # No Gard needed, this just supervise the upsampling

                    # remove feat
                    li_features = li_features * correspondence_gate
                    # add original feat back for keeping point
                    li_features = li_features + (correspondence_original_feat_upsample * correspondence_invert_gate)
                    
                    display_dpp_static = True
                    if display_dpp_static and not self.training:
                        # import ipdb; ipdb.set_trace()
                        print(' At i=%i ponder %i points' % (i, (current_routing_gate == 0.).sum().cpu().numpy()))
                        print(' At i=%i keep %i points' % (i, (current_routing_gate == 1.).sum().cpu().numpy()))
                        self.gate_static[i] += (current_routing_gate == 0.).sum().cpu().numpy()
                        self.keep_gate_static[i] += (current_routing_gate == 1.).sum().cpu().numpy()

                    save_dpp_gate_vis = True
                    if save_dpp_gate_vis and not self.training:
                        ###save per frame 
                        print('saving dpp gate vis')
                        import numpy as np 
                        result_dir = pathlib.Path('/root/hantao/CenterPoint-KITTI/dpp_vis')
                        for j in range(batch_size)  :
                            point_saved_path = result_dir / 'layer_gate_vis'
                            os.makedirs(point_saved_path, exist_ok=True)
                            
                            idx = batch_dict['frame_id'][j]
                            gt = batch_dict['gt_boxes'][j].cpu().numpy()
                            idx = str(idx)
                            
                            dpp_enable_points = xyz_input[j][current_routing_gate[j] == 1.].cpu().numpy()
                            dpp_disable_points = xyz_input[j][current_routing_gate[j] == 0.].cpu().numpy()
                            xyz_input_save_points = xyz_input[j].cpu().numpy()
                            
                            dpp_enable_xyz = point_saved_path / ('dpp_enable_points_list_' + ('%s' % idx) + '_dpp_enable_layer_' + ('%s' % i))
                            dpp_disable_xyz = point_saved_path / ('dpp_enable_points_list_' + ('%s' % idx) + '_dpp_disable_layer_' + ('%s' % i))
                            layer_xyz = point_saved_path / ('dpp_enable_points_list_' + ('%s' % idx) + '_layer_xyz_layer_' + ('%s' % i))
                            sample_gt = point_saved_path / ('dpp_enable_points_list_' + ('%s' % idx) + '_gt_layer_' + ('%s' % i))
                            
                            np.save(str(dpp_enable_xyz), dpp_enable_points)
                            np.save(str(dpp_disable_xyz), dpp_disable_points)
                            np.save(str(layer_xyz), xyz_input_save_points)
                            np.save(str(sample_gt), gt)
                else:
                    correspondence_invert_gate = []
                    correspondence_original_feat_upsample = []
                    li_features_for_dpp = []

            elif self.layer_types[i] == 'Vote_Layer': #i=4
                # print(feature_input.shape)
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                encoder_coords.append(torch.cat([center_origin_batch_idx[..., None].float(),centers_origin.view(batch_size, -1, 3)],dim =-1))
            elif self.layer_types[i] == 'PCT_Layer': # final layer
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features = self.SA_modules[i](xyz_input, feature_input, ctr_xyz)
                li_cls_pred = None
            if torch.isnan(li_xyz).sum() > 0:
                raise RuntimeError('Nan in li_xyz!')
            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)  
            dpp_inverted_gates.append(correspondence_invert_gate)
            dpp_upsampled_features.append(correspondence_original_feat_upsample)  
            dpp_original_features.append(li_features_for_dpp)
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
                pass
            else:
                sa_ins_preds.append([])
           
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
        batch_dict['ctr_batch_idx'] = ctr_batch_idx
        
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1]) # shape?
        batch_dict['centers_features'] = center_features

        # check encoder xyzs
        if torch.isnan(centers).sum() > 0:
            raise RuntimeError('Nan in centers!')


        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features # not used later?
        batch_dict['dpp_gates'] = dpp_gates
        batch_dict['dpp_inverted_gates'] = dpp_inverted_gates
        batch_dict['dpp_upsampled_features'] = dpp_upsampled_features
        batch_dict['dpp_original_features'] = dpp_original_features
        
        if display_dpp_static and not self.training:
            print('gate_static(shutdown points): ', self.gate_static)
            print('gate_static(open points): ', self.keep_gate_static)
        
        ###save per frame 
        # if self.model_cfg.SA_CONFIG.get('SAVE_SAMPLE_LIST',False) and not self.training:  
        #     import numpy as np 
        #     # result_dir = np.load('/home/yifan/tmp.npy', allow_pickle=True)
        #     result_dir = pathlib.Path('/root/dj/code/CenterPoint-KITTI/output/sample_result_radar')
        #     for i in range(batch_size)  :
        #         # i=0      
        #         # point_saved_path = '/home/yifan/tmp'
        #         point_saved_path = result_dir / 'sample_list_save'
        #         os.makedirs(point_saved_path, exist_ok=True)
        #         idx = batch_dict['frame_id'][i]
        #         xyz_list = []
        #         gt = batch_dict['gt_boxes'][i].cpu().numpy()
        #         for sa_xyz in encoder_xyz:
        #             xyz_list.append(sa_xyz[i].cpu().numpy()) 
        #         idx = str(idx)
        #         if '/' in idx: # Kitti_tracking
        #             sample_xyz = point_saved_path / idx.split('/')[0] / ('sample_list_' + ('%s' % idx.split('/')[1]) + '_xyz')
        #             sample_gt = point_saved_path / idx.split('/')[0] / ('sample_list_' + ('%s' % idx.split('/')[1]) + '_gt')
        #             os.makedirs(point_saved_path / idx.split('/')[0], exist_ok=True)

        #         else:
        #             sample_xyz = point_saved_path / ('sample_list_' + ('%s' % idx) + '_xyz')
        #             sample_gt = point_saved_path / ('sample_list_' + ('%s' % idx) + '_gt')
                
        #         np.save(str(sample_gt), gt)
        #         np.save(str(sample_xyz), xyz_list)

                # np.save(str(new_file), point_new.detach().cpu().numpy())
        
        return batch_dict

# class DynamicPointPonderRouterMLP(nn.Module):
#     def __init__(self, feat_num):

#         super(DynamicPointPonderRouterMLP, self).__init__()
#         self.feat_num = feat_num

#         self.gate1 = nn.Linear(self.feat_num, int(self.feat_num/2))
#         self.gate2 = nn.Linear(int(self.feat_num/2), int(self.feat_num/4))
#         self.gate3 = nn.Linear(int(self.feat_num/4), 2)
        
#     def forward(self, x):
#         x = x.permute(0,2,1)
#         x = F.relu(self.gate1(x))
#         x = F.relu(self.gate2(x))
#         x = F.relu(self.gate3(x))
#         return x


class DynamicPointPonderRouterMLP(nn.Module):
    def __init__(self, feat_num):

        super(DynamicPointPonderRouterMLP, self).__init__()
        self.feat_num = feat_num

        self.gate1 = nn.Linear(self.feat_num, self.feat_num)
        self.gate2 = nn.Linear(self.feat_num, 1)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.gate1(x)
        x = self.gate2(x)
        x = torch.stack([x, -x], dim=-1)
        x = F.gumbel_softmax(x, dim=-1, hard=True)[..., 0]
        return x


class DynamicPointPonderUpSampling(nn.Module):
    def __init__(self, input_feat_num, output_feat_num):

        super(DynamicPointPonderUpSampling, self).__init__()
        self.input_feat_num = input_feat_num
        self.output_feat_num = output_feat_num

        self.up1 = nn.Linear(input_feat_num, input_feat_num)
        self.up2 = nn.Linear(input_feat_num, output_feat_num)
        self.up3 = nn.Linear(output_feat_num, output_feat_num)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.activation(x)
        x = x.permute(0,2,1)
        return x
    

class DynamicPointPonderConvUpSampling(nn.Module):
    def __init__(self, mlps, input_feat_num, output_feat_num):

        super(DynamicPointPonderConvUpSampling, self).__init__()
        self.mlps = [j for i in mlps for j in i]
        self.mlps = [input_feat_num] + self.mlps + [output_feat_num]
        shared_mlps = []
        for k in range(len(self.mlps) - 1):
            shared_mlps.extend([
                nn.Conv2d(self.mlps[k], self.mlps[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(self.mlps[k + 1]),
                nn.ReLU()
            ])
        self.up_sample = nn.Sequential(*shared_mlps)
        

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.up_sample(x)
        x = x.squeeze(-1)
        return x