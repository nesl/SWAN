from mmdet3d.registry import MODELS

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from projects.UniM2AE.sst.sst_basic_block_v2 import BasicShiftBlockV2

from projects.UniM2AE.sst.sst_ops import flat2window_v2, window2flat_v2, get_inner_win_inds, get_flat2win_inds_v2, get_window_coors




@MODELS.register_module()
class SSTv2(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        num_attached_conv=2,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        layer_cfg=dict(),
        masked=False,
        ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )

        self.block_list = nn.ModuleList(block_list)
            
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug

        self.masked = masked

        self.num_attached_conv = num_attached_conv

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                    )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                # delattr(convnormrelu[1], 'fp16_enabled')
                conv_list.append(convnormrelu)
            
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, voxel_info):
        '''
        '''
        num_shifts = 2 
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        device = voxel_info['voxel_coors'].device
        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.block_list):
            output = block(output, pos_embed_list, ind_dict_list, 
                padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)

        # If masked we want to send the output to the decoder and not a FPN how requires dense bev image
        if not self.masked:
            output, indices_back, _ = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)

            if self.num_attached_conv > 0:
                for conv in self.conv_layer:
                    output = conv(output)

            output_list = []
            output_list.append(output)
            return output_list
        else:
            if self.num_attached_conv != 0:
                if len(self.output_shape) == 2:
                    bev_out, indices_back, batch_masks = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
                else:
                    bev_out, indices_back, batch_masks = self.recover_volume(output, voxel_info['voxel_coors'], batch_size)
                
                if self.num_attached_conv > 0:
                    for conv in self.conv_layer:
                        bev_out = conv(bev_out)
                voxel_info["output"] = bev_out
                
                return voxel_info, indices_back, batch_masks, output.shape[0]

            voxel_info["output"] = output
            return voxel_info
        
    def recover_volume(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, nx, ny, nz]
        '''
        nx, ny, nz = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny * nz,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 3] * ny * nz + this_coors[:, 2] * nz + this_coors[:, 1]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)
            indices_back.append(indices)
            batch_masks.append(batch_mask)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, nx, ny, nz)

        return batch_canvas, indices_back, batch_masks

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)
            indices_back.append(indices)
            batch_masks.append(batch_mask)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas, indices_back, batch_masks

    
    def recover_bev_indices(self, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        _, nx = self.output_shape

        indices_back = []
        batch_masks = []
        for batch_itt in range(batch_size):
            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)

            indices_back.append(indices)
            batch_masks.append(batch_mask)

        return indices_back, batch_masks



@MODELS.register_module()
class SSTv2Decoder(SSTv2):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        debug=True,
        in_channel=None,
        checkpoint_blocks=[],
        layer_cfg=dict(),
        use_fake_voxels=True,
        num_attached_conv=0,
        conv_in_channel=-1,
        conv_out_channel=-1,
        conv_kwargs=None,
        ):

        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_blocks=num_blocks,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            output_shape=output_shape,
            num_attached_conv=num_attached_conv,
            debug=debug,
            checkpoint_blocks=checkpoint_blocks,
            layer_cfg=layer_cfg,
            masked=True,
            conv_in_channel=conv_in_channel,
            conv_out_channel=conv_out_channel,
            conv_kwargs=conv_kwargs
            )
        

        if in_channel is not None:
            self.enc2dec_projection = nn.Linear(in_channel, d_model[0])
        self._reset_parameters()

        self.use_fake_voxels = use_fake_voxels
        self.mask_token = nn.Parameter(torch.zeros(1, d_model[0]))
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, voxel_info):
        '''
        '''
        voxel_info_encoder = voxel_info
        voxel_info_decoder = voxel_info["voxel_info_decoder"]

        # _____ add in encoder output to the input _____
        encoder_out = voxel_info_encoder["output"]

        # if in_channel project encoder output to right dimension
        if hasattr(self, 'enc2dec_projection'):
            encoder_out = self.enc2dec_projection(encoder_out)

        # replace unmasked voxels with encoder value
        dec2enc_idx = voxel_info_decoder["dec2enc_idx"]
        voxel_feat = voxel_info_decoder['voxel_feats']
        voxel_feat[dec2enc_idx] = encoder_out
        assert torch.allclose(voxel_info_decoder['voxel_coors'][dec2enc_idx], voxel_info_encoder["voxel_coors"]), \
            "Mapping dec2enc not valid"

        # replace masked voxels with masking token
        dec2masked_idx = voxel_info_decoder["dec2masked_idx"]
        n_masked = voxel_info_decoder["n_masked"]
        masked_tokens = self.mask_token.repeat(n_masked, 1)
        voxel_feat[dec2masked_idx] = masked_tokens
        if self.use_fake_voxels:
            # replace fake voxels with masking token
            dec2fake_idx = voxel_info_decoder["dec2fake_idx"]
            n_fake = voxel_info_decoder["n_fake"]
            masked_tokens = self.mask_token.repeat(n_fake, 1)
            voxel_feat[dec2fake_idx] = masked_tokens
        voxel_info_decoder['voxel_feats'] = voxel_feat

        if self.debug:
            test_mapping = -torch.ones(len(voxel_feat), device=voxel_feat.device)
            test_mapping[dec2enc_idx] = 0
            test_mapping[dec2masked_idx] = 1
            if self.use_fake_voxels:
                test_mapping[dec2fake_idx] = 0
            assert not (test_mapping == -1).any(), "All voxels are not covered by the enc_idx and masked_idx"
            assert test_mapping.sum() == n_masked, \
                f"The number of masked voxels differ {test_mapping.sum()} vs {n_masked}"
            assert (1-test_mapping).sum() == len(voxel_feat)-n_masked, \
                f"The number of unmasked voxels differ  {(1-test_mapping).sum()} vs {len(voxel_feat)-n_masked}"
            assert (test_mapping[voxel_info_decoder["dec2input_idx"]].long() == voxel_info_decoder["mask"].long()
                    ).all(), "The masking of the mismatches"

        voxel_info_decoder = super().forward(voxel_info_decoder)

        return voxel_info, voxel_info_decoder, voxel_info_encoder
    

@MODELS.register_module()
class SSTInputLayerV2(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        window_shape,
        sparse_shape,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        ):
        super().__init__()
        self.fp16_enabled = False
        self.meta_drop_info = drop_info
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.mute = mute


    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        original_index = torch.arange(len(voxel_feats), device=voxel_feats.device)

        self.set_drop_info()
        voxel_coors = voxel_coors.long()

        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]
            original_index = original_index[shuffle_inds]

        voxel_info = self.window_partition(voxel_coors)
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors
        voxel_info['original_index'] = original_index
        voxel_info = self.drop_voxel(voxel_info, 2) # voxel_info is updated in this function

        voxel_feats = voxel_info['voxel_feats']  # after dropping
        voxel_coors = voxel_info['voxel_coors']
        original_index = voxel_info['voxel_coors']

        for i in range(2):
            # Dict where for each drop level we give a index to each token
            # unique to all tokens in all windows of that drop level
            voxel_info[f'flat2win_inds_shift{i}'] = \
                get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'], self.drop_info, debug=True)

            # Same structure as above. Positional embedding is done using Sine-Cos embedding within a window, i.e.
            # not related to global position. Position in window is thus x_coord % windows_size_x and same for y
            voxel_info[f'pos_dict_shift{i}'] = \
                self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'], voxel_feats.size(1), voxel_feats.dtype)

            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(voxel_info[f'flat2win_inds_shift{i}'])

        if self.debug:
            coors_3d_dict_shift0 = flat2window_v2(voxel_coors, voxel_info['flat2win_inds_shift0'])
            coors_2d = window2flat_v2(coors_3d_dict_shift0, voxel_info['flat2win_inds_shift0'])
            assert (coors_2d == voxel_coors).all()

        if self.shuffle_voxels:
            voxel_info['shuffle_inds'] = shuffle_inds
        
        return voxel_info
    
    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    def drop_voxel(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.

        Separates windows by the number of tokens e.g.:
        group 1: 0-30 tokens, pad windows with less than 30 token to 30 token
        group 2: 30-60 tokens, pad windows with less than 60 tokens to 60 tokens
        group 3: 60-100000 tokens, pad windows with less than 100 token to 100 tokens
                    drop tokens in windows with more than 100 tokens so that they have 100 tokens
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_info['voxel_coors'])
        voxel_info['voxel_feats'] = voxel_info['voxel_feats'][voxel_keep_inds]
        voxel_info['voxel_coors'] = voxel_info['voxel_coors'][voxel_keep_inds]
        voxel_info['original_index'] = voxel_info['original_index'][voxel_keep_inds]

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        ### sanity check
        if self.debug and self.training:
            for dl in self.drop_info:
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors):
        voxel_info = {}
        for i in range(2):
            # Adds indexation which window (counted across all batches) and which spot in the window
            batch_win_inds, coors_in_win = get_window_coors(coors, self.sparse_shape, self.window_shape, i == 1)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win
        
        return voxel_info

    @torch.no_grad()
    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim, dtype):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''

        # [N,]
        window_shape = self.window_shape
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack([embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)
        
        gap = feat_dim - pos_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((pos_embed_2d.size(0), gap), dtype=dtype, device=coors_in_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        pos_embed_dict = flat2window_v2(
            pos_embed_2d, inds_dict)

        return pos_embed_dict

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(ind_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')


@MODELS.register_module()
class SSTInputLayerV2Masked(SSTInputLayerV2):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        window_shape,
        sparse_shape,
        voxel_size,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        masking_ratio=0.7,
        drop_points_th=100,
        pred_dims=3,
        fake_voxels_ratio=0.1,
        use_chamfer=True,
        use_num_points=True,
        use_fake_voxels=True,
        masked_window_shape=None
        ):
        super().__init__(
            drop_info,
            window_shape,
            sparse_shape,
            shuffle_voxels=shuffle_voxels,
            debug=debug,
            normalize_pos=normalize_pos,
            pos_temperature=pos_temperature,
            mute=mute,
        )
        self.masking_ratio = masking_ratio
        self.drop_points_th = drop_points_th
        self.pred_dims = pred_dims
        self.voxel_size = voxel_size
        self.fake_voxel_ratio = fake_voxels_ratio
        self.use_chamfer = use_chamfer
        self.use_num_points = use_num_points
        self.use_fake_voxels = use_fake_voxels
        self.masked_window_shape = masked_window_shape
        self.unmasked_window_shape = window_shape
        assert use_chamfer or use_num_points or use_fake_voxels, \
            "Need to use at least one of chamfer, num_points, and fake_voxels"

    def forward(self, voxel_feats, voxel_coors, low_level_point_feature, point_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            voxel_coors: shape=[N, 4], [b, z, y, x], voxel coordinate for each voxel
            low_level_point_feature: shape=[Np, 10] [x, y, z, I, cl_x, cl_y, cl_z, ce_x, ce_y, ce_z],
             Np is the point num in the batch, cl_* and ce_*  is position relative to the cluster and center resp.
            point_coors: shape=[Np, 4], [b, z, y, x], voxel coordinate for each point
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        batch_size = int(voxel_coors[:, 0].max().item()) + 1
        device = voxel_feats.device

        gt_dict, fake_voxel_coors = self.get_ground_truth(
            batch_size, device, low_level_point_feature, point_coors, voxel_coors, voxel_feats)

        voxel_info_decoder, voxel_info_encoder = self.mask_voxels(device, voxel_coors, voxel_feats, fake_voxel_coors)

        voxel_info_decoder["gt_dict"] = gt_dict
        voxel_info_encoder["voxel_info_decoder"] = voxel_info_decoder

        return voxel_info_encoder

    def mask_voxels(self, device, voxel_coors, voxel_feats, fake_voxel_coors):
        # Masking voxels: True -> masked, False -> unmasked
        mask = torch.rand(len(voxel_feats), device=device) < self.masking_ratio
        masked_idx, unmasked_idx = mask.nonzero().ravel(), (~mask).nonzero().ravel()
        n_masked_voxels, n_unmasked_voxels = len(masked_idx), len(unmasked_idx)

        # Add fake voxels
        if self.use_fake_voxels:
            fake_voxel_idx = torch.arange(len(fake_voxel_coors), device=device) + len(voxel_coors)
            voxel_coors = torch.cat([voxel_coors, fake_voxel_coors], dim=0)
            fake_voxel_feats = torch.zeros(
                (len(fake_voxel_coors), voxel_feats.shape[1]), device=device, dtype=voxel_feats.dtype)
            voxel_feats = torch.cat([voxel_feats, fake_voxel_feats], dim=0)
            mask = torch.cat([mask, torch.zeros(len(fake_voxel_coors), device=device, dtype=torch.bool)])
            n_fake_voxels = len(fake_voxel_idx)

        # Get info for decoder input, Might drop voxels
        voxel_info_decoder = super().forward(voxel_feats, voxel_coors, batch_size=None)
        assert len(voxel_info_decoder["voxel_feats"]) == len(voxel_feats), "Dropping is not allowed for reconstruction"
        unmasked_voxels = voxel_feats[unmasked_idx]
        unmasked_voxel_coors = voxel_coors[unmasked_idx]

        if self.masked_window_shape is not None:
            # Change window size for masked encoder
            assert self.unmasked_window_shape is not None, "Cannot restore the window shape"
            self.window_shape = self.masked_window_shape
        # Get info for encoder input, Might drop voxels
        voxel_info_encoder = super().forward(unmasked_voxels, unmasked_voxel_coors, batch_size=None)
        assert len(voxel_info_encoder["voxel_feats"]) == n_unmasked_voxels, "Dropping is not allowed for reconstruction"
        if self.masked_window_shape:
            # Change back the window size for the next iteration
            self.window_shape = self.unmasked_window_shape

        voxel_info_decoder["mask"] = mask
        voxel_info_decoder["n_unmasked"] = n_unmasked_voxels
        voxel_info_decoder["n_masked"] = n_masked_voxels
        voxel_info_decoder["unmasked_idx"] = unmasked_idx
        voxel_info_decoder["masked_idx"] = masked_idx
        if self.use_fake_voxels:
            voxel_info_decoder["fake_voxel_idx"] = fake_voxel_idx
            voxel_info_decoder["n_fake"] = n_fake_voxels

        # Index mapping from decoder output to other
        dec2dec_input_idx = torch.argsort(voxel_info_decoder["original_index"])
        dec2masked_idx = dec2dec_input_idx[masked_idx]
        dec2unmasked_idx = dec2dec_input_idx[unmasked_idx]
        dec2enc_idx = dec2unmasked_idx[voxel_info_encoder["original_index"]]
        voxel_info_decoder["dec2input_idx"] = dec2dec_input_idx
        voxel_info_decoder["dec2unmasked_idx"] = dec2unmasked_idx
        voxel_info_decoder["dec2masked_idx"] = dec2masked_idx
        voxel_info_decoder["dec2enc_idx"] = dec2enc_idx
        if self.use_fake_voxels:
            dec2fake_idx = dec2dec_input_idx[fake_voxel_idx]
            voxel_info_decoder["dec2fake_idx"] = dec2fake_idx

        # Debug - sanity check
        if self.debug:
            decoder_feats = voxel_info_decoder["voxel_feats"]
            decoder_coors = voxel_info_decoder["voxel_coors"]

            encoder_feats = voxel_info_encoder["voxel_feats"]
            encoder_coors = voxel_info_encoder["voxel_coors"]

            assert torch.allclose(decoder_feats[dec2dec_input_idx], voxel_feats), \
                "The mapping from decoder to decoder input is invalid"
            assert torch.allclose(decoder_coors[dec2dec_input_idx], voxel_coors.long()), \
                "The mapping from decoder to decoder input is invalid"

            assert torch.allclose(decoder_feats[dec2masked_idx], voxel_feats[masked_idx]), \
                "The mapping from decoder to masked input is invalid"
            assert torch.allclose(decoder_coors[dec2masked_idx], voxel_coors[masked_idx].long()), \
                "The mapping from decoder to masked input is invalid"

            assert torch.allclose(decoder_feats[dec2unmasked_idx], unmasked_voxels), \
                "The mapping from decoder to encoder input is invalid"
            assert torch.allclose(decoder_coors[dec2unmasked_idx], unmasked_voxel_coors.long()), \
                "The mapping from decoder to encoder input is invalid"

            assert torch.allclose(decoder_feats[dec2enc_idx], encoder_feats), \
                "The mapping from decoder to encoder output is invalid"
            assert torch.allclose(decoder_coors[dec2enc_idx], encoder_coors.long()), \
                "The mapping from decoder to encoder output is invalid"

            if self.use_fake_voxels:
                assert (decoder_feats[dec2fake_idx] == 0).all(), \
                    "The mapping from decoder to fake voxels is invalid"
                assert torch.allclose(decoder_coors[dec2fake_idx], fake_voxel_coors.long()), \
                    "The mapping from decoder to fake voxels is invalid"

        return voxel_info_decoder, voxel_info_encoder

    def get_ground_truth(self, batch_size, device, low_level_point_feature, point_coors, voxel_coors, voxel_feats):
        gt_dict = {}
        vx, vy, vz = self.sparse_shape
        max_num_voxels = batch_size * vx * vy * vz

        point_indices = self.get_voxel_indices(point_coors)
        voxel_indices = self.get_voxel_indices(voxel_coors)

        # Points per voxel
        if self.use_num_points:
            n_points_per_voxel_with_zeros = torch.bincount(point_indices)
            point_indices_unique = n_points_per_voxel_with_zeros.nonzero().ravel()
            n_points_per_voxel = n_points_per_voxel_with_zeros[voxel_indices]
            gt_dict["num_points_per_voxel"] = n_points_per_voxel
            assert (n_points_per_voxel > 0).all(), "Exists voxel without connected points"
            assert len(point_indices_unique) == len(voxel_indices), \
                "There is a mismatch between point indices and voxel indices"
            assert (point_indices_unique == voxel_indices.sort()[0]).all(), \
                "There is a mismatch between point indices and voxel indices"

        # Get points per voxel
        if self.use_chamfer:
            points_rel_center = low_level_point_feature[:, -3:]
            assert self.pred_dims in [2, 3], "Either use x and y or x, y, and z"
            points_rel_center = points_rel_center[:, :self.pred_dims].clone()
            pointr_rel_norm = 2 / torch.tensor(self.voxel_size, device=device).view(1, -1)
            points_rel_center = points_rel_center * pointr_rel_norm  # x,y,z all in range [-1, 1]

            shuffle = torch.argsort(torch.rand(len(point_indices)))  # Shuffle to drop random points
            restore = torch.argsort(shuffle)
            inner_voxel_inds = get_inner_win_inds(point_indices[shuffle])[restore]  # fixes one index per point per voxel
            drop_mask = inner_voxel_inds < self.drop_points_th

            points_rel_center = points_rel_center[drop_mask]
            inner_voxel_inds = inner_voxel_inds[drop_mask].long()
            dropped_point_indices = point_indices[drop_mask].long()

            gt_points = torch.zeros((max_num_voxels, self.drop_points_th, 3), device=device, dtype=points_rel_center.dtype)
            gt_points_padding = torch.ones((max_num_voxels, self.drop_points_th), device=device, dtype=torch.long)
            gt_points[dropped_point_indices, inner_voxel_inds] = points_rel_center
            gt_points_padding[dropped_point_indices, inner_voxel_inds] = 0  # not_padded -> 0, padded -> 1
            gt_dict["points_per_voxel"] = gt_points[voxel_indices]
            gt_dict["points_per_voxel_padding"] = gt_points_padding[voxel_indices]
            gt_dict["gt_points"] = low_level_point_feature[:, :self.pred_dims]  # For visualization
            gt_dict["gt_point_coors"] = point_coors  # For visualization

            assert len(gt_dict["points_per_voxel"]) == len(voxel_feats), "Wrong number of point collections"

        if self.use_chamfer and self.use_num_points:
            test_mask = n_points_per_voxel < self.drop_points_th
            _n_points_per_voxel = (1-gt_dict["points_per_voxel_padding"]).sum(1)
            assert (_n_points_per_voxel[test_mask] == n_points_per_voxel[test_mask]).all(), \
                "Mismatch between counted points per voxel and found points per voxel"
            assert (_n_points_per_voxel[~test_mask] == self.drop_points_th).all(), \
                "Error when dropping points for voxels with to many points"

        # TODO: Potentially add fake voxels
        fake_voxel_coors = None
        if self.use_fake_voxels:
            max_num_voxels_per_batch = vx*vy*vz
            voxels_per_batch = torch.bincount(voxel_coors[:, 0].long())  # e.g [5000, 6020, 4920, 5107] for batch_size=4
            n_fake_voxels_per_batch = (voxels_per_batch * self.fake_voxel_ratio).long()  # e.g [500, 602, 492, 510] for fake_voxel_ratio=0.1
            n_fake_voxels = int(n_fake_voxels_per_batch.sum())  # e.g 2104

            occupied = torch.zeros(max_num_voxels, device=device, dtype=torch.bool)
            occupied[voxel_indices] = True
            occupied = occupied.view(batch_size, -1)

            fake_voxel_idxs = [torch.where(~occupied[b])[0]+(b*max_num_voxels_per_batch) for b in range(batch_size)]
            fake_voxel_idxs = torch.cat([
                idx[torch.randperm(len(idx), device=device)][:n_fake]  # Shuffle and take n_fake first indices
                for i, (idx, n_fake) in enumerate(zip(fake_voxel_idxs, n_fake_voxels_per_batch))
            ])
            fake_voxel_coors = torch.zeros((n_fake_voxels, voxel_coors.shape[1]), device=device, dtype=voxel_coors.dtype)
            fake_voxel_coors[:, 0] = fake_voxel_idxs // (vz * vy * vx)  # batch index
            fake_voxel_coors[:, 1] = (fake_voxel_idxs % (vz * vy * vx)) // (vy * vx)  # z index
            fake_voxel_coors[:, 2] = (fake_voxel_idxs % (vy * vx)) // vx  # y index
            fake_voxel_coors[:, 3] = fake_voxel_idxs % vx  # x index

            mask = torch.zeros((len(voxel_coors)+len(fake_voxel_coors)), device=device, dtype=torch.float)
            mask[len(voxel_coors):] = 1.
            gt_dict["fake_voxel_mask"] = mask

        return gt_dict, fake_voxel_coors

    def get_voxel_indices(self, coors):
        vx, vy, vz = self.sparse_shape
        indices = (
                coors[:, 0] * vz * vy * vx +  # batch
                coors[:, 1] * vy * vx +  # z
                coors[:, 2] * vx +  # y
                coors[:, 3]  # x
        ).long()
        return indices


from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch.nn import functional as F



class DynamicVFELayer(nn.Module):
    """Replace the Voxel Feature Encoder layer in VFE layers.

    This layer has the same utility as VFELayer above

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
                 ):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.

        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = F.relu(x)
        return pointwise


@MODELS.register_module()
class DynamicVFE_New(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_gt_points=False
                 ):
        super(DynamicVFE_New, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.return_gt_points = return_gt_points
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).[Nx4] [b,z,y,x]
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # features.requires_grad = True # for cam vis
        # features.register_hook(append_grad(-1)) # for cam vis
        # point_feat_dict[-1] = features.detach() # for cam vis

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            # point_feats.register_hook(append_grad(i)) # for cam vis
            # point_feat_dict[i] = point_feats.detach() # for cam vis

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        if self.return_gt_points:
            return voxel_feats, voxel_coors, low_level_point_feature, coors
        return voxel_feats, voxel_coors