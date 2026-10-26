import os

_CORE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORLDENGINE_ROOT = os.environ.get('WORLDENGINE_ROOT', _CORE)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'{WORLDENGINE_ROOT}/experiments/posttrain/waymo_090_ilft_adagent_smoke'
load_from = f'{WORLDENGINE_ROOT}/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth'
resume_from = None
workflow = [('train', 1)]
custom_imports = dict(
    imports=['mmdet3d_plugin', 'mmdet3d_plugin.datasets.waymo_ilft_dataset'])
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = [
    'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
    'czone_sign', 'generic_object'
]
vehicle_id_list = [0, 1]
group_id_list = [[0], [1], [2], [3, 4, 5, 6]]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = 512
_dim_half_ = 128
canvas_size = (200, 200)
queue_length = 4
past_steps = 3
fut_steps = 4
planning_steps = 8
use_col_optim = False
train_gt_iou_threshold = 0.3
train_dataset_type = 'NavSimOpenSceneE2EFineTune'
dataset_type = 'NavSimOpenSceneE2E'
file_client_args = dict(backend='disk')
data_root = ''
info_root = f'{WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/'
img_root_train = ''
img_root_test = ''
ann_file_train = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl'
ann_file_val = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl'
ann_file_test = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl'
nav_filter_path_train = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_train_tokens.yaml'
nav_filter_path_val = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_val_tokens.yaml'
nav_filter_path_test = f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_val_tokens.yaml'
finetune_yaml = [
    f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_train_tokens.yaml'
]
model = dict(
    type='NAVFormer',
    gt_iou_threshold=0.3,
    queue_length=4,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=7,
    vehicle_id_list=[0, 1],
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    lora_finetuning=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1),
    mem_args=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)),
    pts_bbox_head=dict(
        type='BEVFormerTrackHead',
        bev_h=200,
        bev_w=200,
        num_query=900,
        num_classes=7,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=3,
        fut_steps=4,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            num_cams=8,
            fix_temporal_shift=True,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            num_cams=8,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=4),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=7),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    planning_head=dict(
        type='TrajScoringHead',
        reward_shaping=False,
        use_lora=True,
        trans_use_lora=True,
        rl_finetuning=False,
        importance_sampling=True,
        num_poses=40,
        d_ffn=1024,
        d_model=256,
        vocab_path=
        f'{WORLDENGINE_ROOT}/data/alg_engine/test_8192_kmeans.npy',
        nhead=8,
        nlayers=1,
        num_commands=4,
        transformer_decoder=dict(
            type='BEVOnlyMotionTransformerDecoder',
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            embed_dims=256,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,
                use_lora=True,
                lora_rank=16,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',
                        num_steps=8,
                        embed_dims=256,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1,
                        use_lora=True,
                        lora_rank=16)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        bev_h=200,
        bev_w=200),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesWithDownsample',
        to_float32=True,
        img_root='',
        downsample_factor=2),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
            'czone_sign', 'generic_object'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'sdc_planning',
            'sdc_planning_mask', 'command', 'sdc_planning_world',
            'sdc_planning_past', 'sdc_planning_mask_past',
            'gt_pre_command_sdc', 'sdc_status'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=dict(backend='disk'),
        img_root=''),
    dict(type='ScaleMultiViewImage3D', scale=0.5),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                    'barrier', 'czone_sign', 'generic_object'
                ],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'sdc_planning',
                    'sdc_planning_mask', 'command', 'sdc_planning_world',
                    'sdc_planning_past', 'sdc_planning_mask_past',
                    'gt_pre_command_sdc', 'sdc_status'
                ])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='WaymoOpenSceneE2EFineTune',
        file_client_args=dict(backend='disk'),
        data_root='',
        ann_file=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl',
        nav_filter_path=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_train_tokens.yaml',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesWithDownsample',
                to_float32=True,
                img_root='',
                downsample_factor=2),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                    'barrier', 'czone_sign', 'generic_object'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'sdc_planning',
                    'sdc_planning_mask', 'command', 'sdc_planning_world',
                    'sdc_planning_past', 'sdc_planning_mask_past',
                    'gt_pre_command_sdc', 'sdc_status'
                ])
        ],
        classes=[
            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
            'czone_sign', 'generic_object'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        use_valid_flag=True,
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        queue_length=4,
        past_steps=3,
        fut_steps=4,
        planning_steps=8,
        load_interval=100,
        box_type_3d='LiDAR',
        fix_can_bus_rotation=True,
        finetune_yaml=[
            f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_train_tokens.yaml'
        ],
        normal_ratio=0),
    val=dict(
        type='WaymoOpenSceneE2E',
        file_client_args=dict(backend='disk'),
        data_root='',
        test_mode=True,
        use_valid_flag=True,
        ann_file=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl',
        nav_filter_path=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_val_tokens.yaml',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesInCeph',
                to_float32=True,
                file_client_args=dict(backend='disk'),
                img_root=''),
            dict(type='ScaleMultiViewImage3D', scale=0.5),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1920, 1080),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                            'barrier', 'czone_sign', 'generic_object'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                            'sdc_planning', 'sdc_planning_mask', 'command',
                            'sdc_planning_world', 'sdc_planning_past',
                            'sdc_planning_mask_past', 'gt_pre_command_sdc',
                            'sdc_status'
                        ])
                ])
        ],
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        past_steps=3,
        fut_steps=4,
        classes=[
            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
            'czone_sign', 'generic_object'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        eval_mod=[],
        planning_steps=8,
        fix_can_bus_rotation=True),
    test=dict(
        type='WaymoOpenSceneE2E',
        file_client_args=dict(backend='disk'),
        data_root='',
        test_mode=True,
        use_valid_flag=True,
        ann_file=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_all_infos.pkl',
        nav_filter_path=
        f'{WORLDENGINE_ROOT}/data/custom_scenes/waymo_090/posttrain_rolling/annotations/waymo_090_val_tokens.yaml',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesInCeph',
                to_float32=True,
                file_client_args=dict(backend='disk'),
                img_root=''),
            dict(type='ScaleMultiViewImage3D', scale=0.5),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1920, 1080),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                            'barrier', 'czone_sign', 'generic_object'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                            'sdc_planning', 'sdc_planning_mask', 'command',
                            'sdc_planning_world', 'sdc_planning_past',
                            'sdc_planning_mask_past', 'gt_pre_command_sdc',
                            'sdc_status'
                        ])
                ])
        ],
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        past_steps=3,
        fut_steps=4,
        planning_steps=8,
        classes=[
            'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier',
            'czone_sign', 'generic_object'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        eval_mod=[],
        fix_can_bus_rotation=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 1
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFilesInCeph',
            to_float32=True,
            file_client_args=dict(backend='disk'),
            img_root=''),
        dict(type='ScaleMultiViewImage3D', scale=0.5),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1920, 1080),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'vehicle', 'bicycle', 'pedestrian', 'traffic_cone',
                        'barrier', 'czone_sign', 'generic_object'
                    ],
                    with_label=False),
                dict(
                    type='CustomCollect3D',
                    keys=[
                        'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                        'sdc_planning', 'sdc_planning_mask', 'command',
                        'sdc_planning_world', 'sdc_planning_past',
                        'sdc_planning_mask_past', 'gt_pre_command_sdc',
                        'sdc_status'
                    ])
            ])
    ])
runner = dict(type='EpochBasedRunner', max_epochs=1)
find_unused_parameters = True
gpu_ids = range(0, 1)
