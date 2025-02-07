import pathlib
from pyexpat import model
import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os

class IASSD_Backbone_DLP(nn.Module):
    """ Backbone for IA-SSD"""

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class
        

        self.SA_modules = nn.ModuleList()
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

        # pondernet
        self.halting_lambda_layer1 = nn.Linear(2048,1)
        self.halting_lambda_layer2 = nn.Linear(256,1)
        self.halting_max_step = 2

        self.stop_layer_info = [0,0,0]

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
                        use_pooling_weights=self.use_pooling_weight
                    )
                )

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
        encoder_li_cls_pred = []

        # halting related
        halting_p = []

        # execute the 0,1,2 SA layers
        li_cls_pred_temp = []
        li_cls_pred = None
        for i in [0,1,2]:
            # print('SA now in layer for early feature obtain', i)
            # print('getting intput from layer', self.layer_inputs[i])
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

                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz, save_features_dir=save_path, frame_id=batch_dict['frame_id'][0])
                
                # halting realted
                if i == self.halting_max_step:
                    lambda_n = li_features.new_ones(batch_size)
                else:
                    if i == 0:
                        # vectors to save intermediate values
                        un_halted_prob = li_features.new_ones((batch_size,))  # unhalted probability till step n
                        halting_step = li_features.new_zeros((batch_size,), dtype=torch.long)  # stopping step
                    # print(li_features.shape)
                    # print(li_xyz.shape)
                    lambda_n = self.halting_lambda_layer1(li_features)  # (B, C, 1)
                    # print('lambda_n.shape', lambda_n.shape)
                    lambda_n = lambda_n.squeeze(-1)  # (B, C)
                    # print('lambda_n.shape', lambda_n.shape)
                    lambda_n = self.halting_lambda_layer2(lambda_n)
                    # print('lambda_n.shape', lambda_n.shape)
                    lambda_n = torch.sigmoid(lambda_n).squeeze()
                    # print('lambda_n.shape', lambda_n.shape)
                p_n = un_halted_prob * lambda_n
                halting_p.append(p_n)
                # calculate halting step
                halting_step = torch.maximum(
                    i
                    * (halting_step == 0)
                    * torch.bernoulli(lambda_n).to(torch.long),
                    halting_step)
                # track unhalted probability and flip coin to halt
                un_halted_prob = un_halted_prob * (1 - lambda_n)
                # break if we are in inference and all elements have halting_step
                if not self.training and (halting_step > 0).sum() == batch_size:
                    # import ipdb; ipdb.set_trace();
                    self.stop_layer_info[i] = self.stop_layer_info[i] + 1
                    print('Frame', batch_dict['frame_id'][0] ,'Stop Eearly at layer', i ,'! halting_step=', halting_step, 'stop_layer_info=', self.stop_layer_info)
                    break

            if torch.isnan(li_xyz).sum() > 0:
                raise RuntimeError('Nan in li_xyz!')
            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            # print('adding layer', i, 'to li_features.shape' , li_features.shape)
            encoder_features.append(li_features)            
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
                pass
            else:
                sa_ins_preds.append([])
            li_cls_pred_temp.append(li_cls_pred)

        # print('Early feature obtain done')
        # import ipdb; ipdb.set_trace()
        if self.training: 
            post_sa_list = [0,1,2]
        else:
            post_sa_list = [p for p in range(halting_step.cpu().numpy().max())]
            while len(encoder_xyz) != 4:
                encoder_xyz.append(None)
                encoder_features.append(None)
                sa_ins_preds.append([])
                encoder_coords.append(None)
                li_cls_pred_temp.append(None)
        for j in post_sa_list:
            # print('Skipping calculation started. Started from ', j)

            li_cls_pred = li_cls_pred_temp[j]
            for i in range(len(self.SA_modules)):
                if i in [0,1,2]:
                    # skip the first 3 SA layers
                    continue
                # print('Overall exectuion now in layer', i)
                # import ipdb; ipdb.set_trace()
                if self.layer_inputs[i] is 3:
                    # print('redirecting to input layer', j)
                    xyz_input = encoder_xyz[j+1]
                    feature_input = encoder_features[j+1]
                else:
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

                    li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz, save_features_dir=save_path, frame_id=batch_dict['frame_id'][0])
                    # print('index i=', i)
                    # print('feature_input.shape:', feature_input.shape)
                    # print('li_xyz.shape: ', li_xyz.shape)
                    # print('li_features.shape: ', li_features.shape)
                    # print('li_cls_pred.shape: ', li_cls_pred.shape) if li_cls_pred is not None else None

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
                # print('adding layer', i, 'to li_features.shape' , li_features.shape)
                encoder_features.append(li_features)            
                if li_cls_pred is not None:
                    li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                    sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
                    pass
                else:
                    sa_ins_preds.append([])
            
            ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
            batch_dict['%s_ctr_offsets' % j] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
            batch_dict['%s_centers'% j] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
            batch_dict['%s_centers_origin'% j] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
            batch_dict['%s_ctr_batch_idx'% j] = ctr_batch_idx
            
            center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1]) # shape?
            batch_dict['%s_centers_features'% j] = center_features

            # check encoder xyzs
            if torch.isnan(centers).sum() > 0:
                raise RuntimeError('Nan in centers!')

            batch_dict['%s_encoder_xyz'% j] = encoder_xyz
            batch_dict['%s_encoder_coords'% j] = encoder_coords
            batch_dict['%s_sa_ins_preds'% j] = sa_ins_preds
            batch_dict['%s_encoder_features'% j] = encoder_features # not used later?


        batch_dict['halting_p'] = torch.stack(halting_p)
        batch_dict['halting_step'] = halting_step
        
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