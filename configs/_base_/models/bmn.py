model = dict(
    type="BMN",
    projection=dict(
        type="ConvSingleProj",  # ConvSingleProj
        in_channels=2048,
        out_channels=256,
        num_convs=2,
        conv_cfg=dict(groups=4),
    ),
    rpn_head=dict(
        type="TemporalEvaluationHead",  # TemporalEvaluationHead
        in_channels=256,
        num_classes=2,
        conv_cfg=dict(groups=4),
        loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
    ),
    roi_head=dict(
        type="StandardProposalMapHead",  # StandardProposalMapHead
        proposal_generator=dict(type="DenseProposalMap", tscale=125, dscale=125),
        proposal_roi_extractor=dict(
            type="BMNExtractor",
            in_channels=256,
            roi_channels=512,
            out_channels=128,
            tscale=125,
            dscale=125,
            prop_extend_ratio=0.5,
        ),
        proposal_head=dict(
            type="PEMHead",  # FC_head
            in_channels=128,
            feat_channels=128,
            num_convs=2,
            num_classes=2,
            loss=dict(
                cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3, weight=5.0),
            ),
        ),
    ),
)



