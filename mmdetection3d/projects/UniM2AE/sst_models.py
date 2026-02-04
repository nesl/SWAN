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
    # keep layer mask is [0, 1, 0, ...] of length num_layers where 1 is run and 0 is skip
    def forward(self, voxel_info, keep_layer_mask=None):
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
            if keep_layer_mask and not keep_layer_mask[i]: # Do not run block if we keep_layer_mask[i] == 0
                continue
            output = block(output, pos_embed_list, ind_dict_list, 
                padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)

        # If masked we want to send the output to the decoder and not a FPN how requires dense bev image
        if not self.masked:
            # output, indices_back, _ = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)

            # if self.num_attached_conv > 0:
            #     for conv in self.conv_layer:
            #         output = conv(output)

            # output_list = []
            # output_list.append(output)
            # return output_list
            if len(self.output_shape) == 2:
                output, indices_back, _ = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)
            elif len(self.output_shape) == 3:
                output, indices_back, _ = self.recover_volume(output, voxel_info['voxel_coors'], batch_size)
            else:
                raise ValueError("The output_shape should be [H, W, Z] or [H, W]")

            if self.num_attached_conv > 0:
                for conv in self.conv_layer:
                    output = conv(output)

            return output
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

            # Handle empty batches
            if this_coors.shape[0] == 0:
                batch_canvas.append(canvas)
                indices_back.append(torch.tensor([], dtype=torch.long, device=coors.device))
                batch_masks.append(batch_mask)
                continue

            # Validate coordinates are within bounds
            x_coords = this_coors[:, 3]
            y_coords = this_coors[:, 2]
            z_coords = this_coors[:, 1]
            valid_mask = (x_coords >= 0) & (x_coords < nx) & \
                         (y_coords >= 0) & (y_coords < ny) & \
                         (z_coords >= 0) & (z_coords < nz)

            # Save whether we need to filter before modifying tensors
            needs_filtering = not valid_mask.all()

            if needs_filtering:
                x_coords = x_coords[valid_mask]
                y_coords = y_coords[valid_mask]
                z_coords = z_coords[valid_mask]

            # Skip if no valid coordinates remain
            if x_coords.shape[0] == 0:
                batch_canvas.append(canvas)
                indices_back.append(torch.tensor([], dtype=torch.long, device=coors.device))
                batch_masks.append(batch_mask)
                continue

            indices = x_coords * ny * nz + y_coords * nz + z_coords
            indices = indices.type(torch.long)

            # Get corresponding voxel features (apply same valid_mask)
            voxels = voxel_feat[batch_mask, :]
            if needs_filtering:
                voxels = voxels[valid_mask]
            voxels = voxels.t()  # [c, n]

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

            # Handle empty batches
            if this_coors.shape[0] == 0:
                batch_canvas.append(canvas)
                indices_back.append(torch.tensor([], dtype=torch.long, device=coors.device))
                batch_masks.append(batch_mask)
                continue

            # Validate coordinates are within bounds
            y_coords = this_coors[:, 2]
            x_coords = this_coors[:, 3]
            valid_mask = (y_coords >= 0) & (y_coords < ny) & (x_coords >= 0) & (x_coords < nx)

            # Save whether we need to filter before modifying tensors
            needs_filtering = not valid_mask.all()

            if needs_filtering:
                y_coords = y_coords[valid_mask]
                x_coords = x_coords[valid_mask]

            # Skip if no valid coordinates remain
            if y_coords.shape[0] == 0:
                batch_canvas.append(canvas)
                indices_back.append(torch.tensor([], dtype=torch.long, device=coors.device))
                batch_masks.append(batch_mask)
                continue

            indices = y_coords * nx + x_coords
            indices = indices.type(torch.long)

            # Get corresponding voxel features (apply same valid_mask)
            voxels = voxel_feat[batch_mask, :]
            if needs_filtering:
                voxels = voxels[valid_mask]
            voxels = voxels.t()  # [c, n]

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
        
        self.vx, self.vy, self.vz = self.sparse_shape
        self.max_voxels_per_sample = self.vx * self.vy * self.vz

    def forward(self, voxel_feats, voxel_coors, low_level_point_feature, point_coors, batch_size=None):
        if batch_size is None:
            batch_size = int(voxel_coors[:, 0].max().item()) + 1

        device = voxel_feats.device

        # Filter out voxels with coordinates outside valid sparse_shape
        valid_voxel_mask = (
            (voxel_coors[:, 0] >= 0) & (voxel_coors[:, 0] < batch_size) &
            (voxel_coors[:, 1] >= 0) & (voxel_coors[:, 1] < self.vz) &
            (voxel_coors[:, 2] >= 0) & (voxel_coors[:, 2] < self.vy) &
            (voxel_coors[:, 3] >= 0) & (voxel_coors[:, 3] < self.vx)
        )

        if not valid_voxel_mask.all():
            voxel_feats = voxel_feats[valid_voxel_mask]
            voxel_coors = voxel_coors[valid_voxel_mask]

        # Also validate point coordinates
        valid_point_mask = (
            (point_coors[:, 0] >= 0) & (point_coors[:, 0] < batch_size) &
            (point_coors[:, 1] >= 0) & (point_coors[:, 1] < self.vz) &
            (point_coors[:, 2] >= 0) & (point_coors[:, 2] < self.vy) &
            (point_coors[:, 3] >= 0) & (point_coors[:, 3] < self.vx)
        )

        if not valid_point_mask.all():
            low_level_point_feature = low_level_point_feature[valid_point_mask]
            point_coors = point_coors[valid_point_mask]

        gt_dict, fake_voxel_coors = self.get_ground_truth(
            batch_size, device, low_level_point_feature, point_coors, voxel_coors, voxel_feats)

        voxel_info_decoder, voxel_info_encoder = self.mask_voxels(device, voxel_coors, voxel_feats, fake_voxel_coors)

        voxel_info_decoder["gt_dict"] = gt_dict
        voxel_info_encoder["voxel_info_decoder"] = voxel_info_decoder

        return voxel_info_encoder

    def mask_voxels(self, device, voxel_coors, voxel_feats, fake_voxel_coors):
        mask = torch.rand(len(voxel_feats), device=device) < self.masking_ratio
        
        masked_idx = mask.nonzero().flatten()
        unmasked_idx = (~mask).nonzero().flatten()
        
        n_masked_voxels = len(masked_idx)
        n_unmasked_voxels = len(unmasked_idx)

        # Add fake voxels
        if self.use_fake_voxels and fake_voxel_coors is not None:
            valid_fake_mask = (
                (fake_voxel_coors[:, 1] >= 0) & (fake_voxel_coors[:, 1] < self.vz) &
                (fake_voxel_coors[:, 2] >= 0) & (fake_voxel_coors[:, 2] < self.vy) &
                (fake_voxel_coors[:, 3] >= 0) & (fake_voxel_coors[:, 3] < self.vx)
            )

            if not valid_fake_mask.all():
                fake_voxel_coors = fake_voxel_coors[valid_fake_mask]

            n_fake_voxels = len(fake_voxel_coors)

            if n_fake_voxels > 0:
                fake_voxel_idx = torch.arange(n_fake_voxels, device=device) + len(voxel_coors)

                voxel_coors = torch.cat([voxel_coors, fake_voxel_coors], dim=0)

                fake_voxel_feats = torch.zeros(
                    (n_fake_voxels, voxel_feats.shape[1]), device=device, dtype=voxel_feats.dtype)
                voxel_feats = torch.cat([voxel_feats, fake_voxel_feats], dim=0)

                mask_ext = torch.zeros(n_fake_voxels, device=device, dtype=torch.bool)
                mask = torch.cat([mask, mask_ext])
            else:
                n_fake_voxels = 0
        else:
            n_fake_voxels = 0

        voxel_info_decoder = super().forward(voxel_feats, voxel_coors, batch_size=None)
        assert len(voxel_info_decoder["voxel_feats"]) == len(voxel_feats), "Dropping is not allowed for reconstruction"
        unmasked_voxels = voxel_feats[unmasked_idx]
        unmasked_voxel_coors = voxel_coors[unmasked_idx]

        if self.masked_window_shape is not None:
            self.window_shape = self.masked_window_shape
            
        voxel_info_encoder = super().forward(unmasked_voxels, unmasked_voxel_coors, batch_size=None)
        assert len(voxel_info_encoder["voxel_feats"]) == n_unmasked_voxels, "Dropping is not allowed for reconstruction"
        
        if self.masked_window_shape:
            self.window_shape = self.unmasked_window_shape

        voxel_info_decoder["mask"] = mask
        voxel_info_decoder["n_unmasked"] = n_unmasked_voxels
        voxel_info_decoder["n_masked"] = n_masked_voxels
        voxel_info_decoder["unmasked_idx"] = unmasked_idx
        voxel_info_decoder["masked_idx"] = masked_idx

        if self.use_fake_voxels and n_fake_voxels > 0:
            voxel_info_decoder["fake_voxel_idx"] = fake_voxel_idx
            voxel_info_decoder["n_fake"] = n_fake_voxels

        dec2dec_input_idx = torch.argsort(voxel_info_decoder["original_index"])
        dec2masked_idx = dec2dec_input_idx[masked_idx]
        dec2unmasked_idx = dec2dec_input_idx[unmasked_idx]
        dec2enc_idx = dec2unmasked_idx[voxel_info_encoder["original_index"]]
        voxel_info_decoder["dec2input_idx"] = dec2dec_input_idx
        voxel_info_decoder["dec2unmasked_idx"] = dec2unmasked_idx
        voxel_info_decoder["dec2masked_idx"] = dec2masked_idx
        voxel_info_decoder["dec2enc_idx"] = dec2enc_idx

        if self.use_fake_voxels and n_fake_voxels > 0:
            dec2fake_idx = dec2dec_input_idx[fake_voxel_idx]
            voxel_info_decoder["dec2fake_idx"] = dec2fake_idx

        return voxel_info_decoder, voxel_info_encoder

    def get_ground_truth(self, batch_size, device, low_level_point_feature, point_coors, voxel_coors, voxel_feats):
        gt_dict = {}
        # Ensure we allocate enough space
        max_num_voxels_total = batch_size * self.max_voxels_per_sample

        point_indices = self.get_voxel_indices(point_coors)
        voxel_indices = self.get_voxel_indices(voxel_coors)

        # Discard points that fall outside the defined grid
        valid_point_mask = (point_indices >= 0) & (point_indices < max_num_voxels_total)
        point_indices = point_indices[valid_point_mask]
        point_coors = point_coors[valid_point_mask]
        low_level_point_feature = low_level_point_feature[valid_point_mask]

        # Points per voxel (Exact Count)
        if self.use_num_points:
            n_points_per_voxel_with_zeros = torch.bincount(point_indices, minlength=max_num_voxels_total)
            
            # voxel_indices might be OOB if config mismatches data. 
            # map OOB indices to 0, read the value, then mask the result to 0.
            valid_voxel_mask = (voxel_indices >= 0) & (voxel_indices < max_num_voxels_total)
            safe_voxel_indices = voxel_indices * valid_voxel_mask.long() # Map bad indices to 0
            
            gathered_counts = n_points_per_voxel_with_zeros[safe_voxel_indices]
            gathered_counts[~valid_voxel_mask] = 0 # Mask out bad reads
            gt_dict["num_points_per_voxel"] = gathered_counts

        # Chamfer Ground Truth
        if self.use_chamfer:
            points_rel_center = low_level_point_feature[:, -3:]
            pointr_rel_norm = 2 / torch.tensor(self.voxel_size, device=device).view(1, -1)
            points_rel_center = points_rel_center[:, :self.pred_dims] * pointr_rel_norm

            # Linear Shuffle for random sampling
            perm = torch.randperm(len(point_indices), device=device)
            shuffled_point_indices = point_indices[perm]
            
            # Sort is required for get_inner_win_inds to group points by voxel correctly
            sort_order = torch.argsort(shuffled_point_indices)
            sorted_point_indices = shuffled_point_indices[sort_order]
            
            # Calculate inner indices (0..N for points in same voxel)
            inner_voxel_inds = get_inner_win_inds(sorted_point_indices)
            drop_mask = inner_voxel_inds < self.drop_points_th

            # Filter
            final_point_indices = sorted_point_indices[drop_mask]
            final_inner_inds = inner_voxel_inds[drop_mask]
            
            # Map back to permuted data
            final_perm_indices = perm[sort_order][drop_mask]
            final_points_rel = points_rel_center[final_perm_indices]

            # Allocate dense tensors
            gt_points = torch.zeros((max_num_voxels_total, self.drop_points_th, 3), device=device, dtype=points_rel_center.dtype)
            gt_points_padding = torch.ones((max_num_voxels_total, self.drop_points_th), device=device, dtype=torch.long)
            
            # Scatter 
            gt_points[final_point_indices, final_inner_inds] = final_points_rel
            gt_points_padding[final_point_indices, final_inner_inds] = 0

            # Gather for GT Points ---
            valid_voxel_mask = (voxel_indices >= 0) & (voxel_indices < max_num_voxels_total)
            safe_voxel_indices = voxel_indices * valid_voxel_mask.long()

            # Gather points
            gathered_points = gt_points[safe_voxel_indices]
            gathered_points[~valid_voxel_mask] = 0
            gt_dict["points_per_voxel"] = gathered_points

            # Gather padding
            gathered_padding = gt_points_padding[safe_voxel_indices]
            gathered_padding[~valid_voxel_mask] = 1 # Treat OOB as fully padded
            gt_dict["points_per_voxel_padding"] = gathered_padding
            
            gt_dict["gt_points"] = low_level_point_feature[:, :self.pred_dims]
            gt_dict["gt_point_coors"] = point_coors

        fake_voxel_coors = None
        if self.use_fake_voxels:
            occupied = torch.zeros(max_num_voxels_total, device=device, dtype=torch.bool)
            
            # Use valid indices only to mark occupancy
            valid_voxel_mask = (voxel_indices >= 0) & (voxel_indices < max_num_voxels_total)
            valid_indices_for_occ = voxel_indices[valid_voxel_mask]
            occupied[valid_indices_for_occ] = True
            
            # Use minlength to ensure we have counts for all batches
            voxels_per_batch = torch.bincount(voxel_coors[:, 0].long(), minlength=batch_size)
            n_fake_needed_per_batch = (voxels_per_batch * self.fake_voxel_ratio).long()
            
            fake_idxs_list = []
            
            for b in range(batch_size):
                n_needed = n_fake_needed_per_batch[b].item()
                if n_needed == 0: continue
                
                batch_offset = b * self.max_voxels_per_sample
                
                # Oversample 1.5x
                n_sample = int(n_needed * 1.5) + 64
                candidates = torch.randint(0, self.max_voxels_per_sample, (n_sample,), device=device)
                
                global_candidates = candidates + batch_offset
                global_candidates = global_candidates[global_candidates < occupied.shape[0]]
                # Filter valid
                valid_candidates = global_candidates[~occupied[global_candidates]]
                

                # Check unique to prevent duplicates
                valid_candidates = torch.unique(valid_candidates)
                
                if len(valid_candidates) >= n_needed:
                    fake_idxs_list.append(valid_candidates[:n_needed])
                else:
                    # Fallback path
                    fake_idxs_list.append(valid_candidates)
                    remaining = n_needed - len(valid_candidates)
                    
                    batch_occupied = occupied[batch_offset : batch_offset + self.max_voxels_per_sample]
                    all_free = (~batch_occupied).nonzero().flatten() + batch_offset
                    
                    if len(all_free) > 0:
                        # Handle case where remaining > len(all_free) by clamping
                        count = min(remaining, len(all_free))
                        perm = torch.randperm(len(all_free), device=device)[:count]
                        fake_idxs_list.append(all_free[perm])

            if len(fake_idxs_list) > 0:
                fake_voxel_idxs = torch.cat(fake_idxs_list)
                n_fake_voxels = len(fake_voxel_idxs)

                fake_voxel_coors = torch.zeros((n_fake_voxels, voxel_coors.shape[1]), device=device, dtype=voxel_coors.dtype)
                
                fake_voxel_coors[:, 0] = fake_voxel_idxs // self.max_voxels_per_sample
                remainder = fake_voxel_idxs % self.max_voxels_per_sample
                fake_voxel_coors[:, 1] = remainder // (self.vy * self.vx)
                remainder = remainder % (self.vy * self.vx)
                fake_voxel_coors[:, 2] = remainder // self.vx
                fake_voxel_coors[:, 3] = remainder % self.vx

                mask = torch.zeros((len(voxel_coors) + n_fake_voxels), device=device, dtype=torch.float)
                mask[len(voxel_coors):] = 1.
                gt_dict["fake_voxel_mask"] = mask
            else:
                 mask = torch.zeros(len(voxel_coors), device=device, dtype=torch.float)
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


def DynamicScatterMean(features, coors, voxel_size, point_cloud_range, return_inverse=False):
    """
    DynamicScatterMean
    Matches logic of DynamicScatter(mode='avg') but prevents CUDA crashes.
    Associates points to voxels based on coordinates and averages features within each voxel.
    """
    device = features.device
    vx, vy, vz = voxel_size

    # Grid dimensions
    grid_x = round((point_cloud_range[3] - point_cloud_range[0]) / vx)
    grid_y = round((point_cloud_range[4] - point_cloud_range[1]) / vy)
    grid_z = round((point_cloud_range[5] - point_cloud_range[2]) / vz)

    # coordinates are [B, Z, Y, X]
    # In the C kernel: "T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;"
    b_idx = coors[:, 0].long()
    z_idx = torch.clamp(coors[:, 1].long(), 0, grid_z - 1)
    y_idx = torch.clamp(coors[:, 2].long(), 0, grid_y - 1)
    x_idx = torch.clamp(coors[:, 3].long(), 0, grid_x - 1)

    # Flatten for unique grouping
    # Prevent using the full grid
    indices = (b_idx * grid_z * grid_y * grid_x +
               z_idx * grid_y * grid_x +
               y_idx * grid_x +
               x_idx)
    # find unqiue voxels
    unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
    num_voxels = unique_indices.shape[0]
    
    # Aggregate
    # create zero tensor to hold sums
    sum_feats = torch.zeros(num_voxels, features.shape[1], device=device, dtype=features.dtype)
    # sum up features into their respective voxel slots
    sum_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.shape[1]), features)
    # count number of points per voxel
    counts = torch.zeros(num_voxels, device=device, dtype=features.dtype)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(b_idx, dtype=features.dtype))
    
    voxel_feats = sum_feats / counts.clamp(min=1).unsqueeze(1)

    # Back to [B, Z, Y, X]
    voxel_coors = torch.zeros(num_voxels, 4, device=device, dtype=torch.long)
    voxel_coors[:, 0] = unique_indices // (grid_z * grid_y * grid_x)
    rem = unique_indices % (grid_z * grid_y * grid_x)
    voxel_coors[:, 1] = rem // (grid_y * grid_x)
    rem = rem % (grid_y * grid_x)
    voxel_coors[:, 2] = rem // grid_x
    voxel_coors[:, 3] = rem % grid_x

    if return_inverse:
        return voxel_feats, voxel_coors, inverse_indices
    return voxel_feats, voxel_coors


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

    The main difference is that we calculate distance feature with 3d rather than 1d.

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
        # here we use 3d distance instead of 1d distance
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
        # This is never used?
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
            self.fusion_layer = MODELS.build(fusion_layer)

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors, batch_size=None):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point. [N, 4] in [b, z, y, x] order
            voxel_mean (torch.Tensor): Voxel features to be mapped. [M, C]
            voxel_coors (torch.Tensor): Coordinates of valid voxels [M, 4] in [b, z, y, x] order

        Returns:
            torch.Tensor: Features or centers of each point. [N, C]
        """
        # Calculate grid dimensions
        canvas_z = round((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)

        if batch_size is None:
            batch_size = int(pts_coors[:, 0].max().item()) + 1

        # Handle empty inputs
        if voxel_mean.shape[0] == 0 or pts_coors.shape[0] == 0:
            return voxel_mean.new_zeros((pts_coors.shape[0], voxel_mean.shape[1]))

        canvas_len = canvas_z * canvas_y * canvas_x * batch_size

        # Initialize canvas with -1 to indicate empty voxels
        canvas = voxel_mean.new_full((canvas_len,), -1, dtype=torch.long)

        # Clamp voxel coordinates to valid range before computing indices
        v_batch = voxel_coors[:, 0].long()
        v_z = torch.clamp(voxel_coors[:, 1].long(), 0, canvas_z - 1)
        v_y = torch.clamp(voxel_coors[:, 2].long(), 0, canvas_y - 1)
        v_x = torch.clamp(voxel_coors[:, 3].long(), 0, canvas_x - 1)

        # Only include voxels with valid batch indices
        valid_vox_mask = (v_batch >= 0) & (v_batch < batch_size)

        if valid_vox_mask.any():
            v_indices = (v_batch[valid_vox_mask] * canvas_z * canvas_y * canvas_x +
                        v_z[valid_vox_mask] * canvas_y * canvas_x +
                        v_y[valid_vox_mask] * canvas_x +
                        v_x[valid_vox_mask])

            # Clamp indices as final safety
            v_indices = torch.clamp(v_indices, 0, canvas_len - 1)

            # Create index mapping
            voxel_indices = torch.arange(voxel_mean.size(0), device=voxel_mean.device)
            canvas[v_indices] = voxel_indices[valid_vox_mask]

        # Clamp point coordinates to valid range
        p_batch = pts_coors[:, 0].long()
        p_z = torch.clamp(pts_coors[:, 1].long(), 0, canvas_z - 1)
        p_y = torch.clamp(pts_coors[:, 2].long(), 0, canvas_y - 1)
        p_x = torch.clamp(pts_coors[:, 3].long(), 0, canvas_x - 1)

        # Compute point indices (already clamped, so always valid)
        point_indices = (p_batch * canvas_z * canvas_y * canvas_x +
                        p_z * canvas_y * canvas_x +
                        p_y * canvas_x + p_x)

        # Clamp as final safety
        point_indices = torch.clamp(point_indices, 0, canvas_len - 1)

        # Gather voxel indices from canvas
        voxel_inds = canvas[point_indices]

        # Points are valid if they map to an existing voxel and have valid batch
        valid_pts_mask = (voxel_inds >= 0) & (p_batch >= 0) & (p_batch < batch_size)

        # Prepare output
        center_per_point = voxel_mean.new_zeros((pts_coors.shape[0], voxel_mean.shape[1]))

        if valid_pts_mask.any():
            # Clamp voxel_inds to valid range for safety
            safe_voxel_inds = torch.clamp(voxel_inds[valid_pts_mask], 0, voxel_mean.size(0) - 1)
            center_per_point[valid_pts_mask] = voxel_mean[safe_voxel_inds]

        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                batch_size=None):
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
        if batch_size is None:
            batch_size = int(coors[:, 0].max().item()) + 1
        else:
            batch_size = int(batch_size)

        # Calculate grid dimensions
        canvas_z = round((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)

        # Clamp coordinates instead of filtering to preserve tensor shapes for backward pass
        coors_clamped = coors.clone()
        coors_clamped[:, 0] = torch.clamp(coors[:, 0], 0, batch_size - 1)
        coors_clamped[:, 1] = torch.clamp(coors[:, 1], 0, canvas_z - 1)  # z
        coors_clamped[:, 2] = torch.clamp(coors[:, 2], 0, canvas_y - 1)  # y
        coors_clamped[:, 3] = torch.clamp(coors[:, 3], 0, canvas_x - 1)  # x



        features_ls = [features]

        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            # Use DynamicScatterMean with return_inverse for efficient point mapping
            voxel_mean, mean_coors, cluster_inverse = DynamicScatterMean(
                features[:, :3], coors_clamped,
                (self.vx, self.vy, self.vz),
                self.point_cloud_range,
                return_inverse=True
            )
            # Map voxel mean back to points using inverse indices (much faster than canvas lookup)
            points_mean = voxel_mean[cluster_inverse]
            f_cluster = features[:, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center (use clamped coords)
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors_clamped[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors_clamped[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors_clamped[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            features_ls.append(features[:, :3]) # 3 channels (x, y, z)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats, voxel_coors_out = None, None
        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)

            # Use DynamicScatterMean instead of MMCV DynamicScatter to avoid CUDA backward crash
            voxel_feats, voxel_coors_out, vfe_inverse = DynamicScatterMean(
                point_feats, coors_clamped,
                (self.vx, self.vy, self.vz),
                self.point_cloud_range,
                return_inverse=True
            )

            if i != len(self.vfe_layers) - 1:
                # Use inverse indices for efficient point mapping instead of canvas lookup
                feat_per_point = voxel_feats[vfe_inverse]
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
    
        if self.return_gt_points:
            return voxel_feats, voxel_coors_out, low_level_point_feature, coors_clamped

        return voxel_feats, voxel_coors_out