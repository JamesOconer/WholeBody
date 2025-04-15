import torch
import transforms

class MSELoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
        kpts = torch.Tensor([t["keypoints"] for t in targets]).to(device)
        gt_keypoints = kpts[:,0:23,:]
        nose = kpts[:,0,:].unsqueeze(dim=1)
        exceptNose = gt_keypoints - nose
        dist2nose = torch.sqrt(torch.square(exceptNose[:,:,0])+torch.square(exceptNose[:,:,1]))
        leftHip = gt_keypoints[:,11,:].unsqueeze(dim=1)
        exceptLefthip = gt_keypoints - leftHip
        dist2lefthip = torch.sqrt(torch.square(exceptLefthip[:,:,0])+torch.square(exceptLefthip[:,:,1]))
        rightHip = gt_keypoints[:,12,:].unsqueeze(dim=1)
        exceptRighthip = gt_keypoints - rightHip
        dist2righthip = torch.sqrt(torch.square(exceptRighthip[:, :, 0]) + torch.square(exceptRighthip[:, :, 1]))
        distBody = torch.mean(torch.cat((dist2nose, dist2lefthip, dist2righthip)))

        # gt_foot = kpts[:,17:23,:]
        gt_face = kpts[:,23:91,:]
        nosehip = kpts[:,53,:].unsqueeze(dim=1)
        exceptNosehip = gt_face - nosehip
        dist2nosehip = torch.sqrt(torch.square(exceptNosehip[:,:,0])+torch.square(exceptNosehip[:,:,1]))
        distFace = torch.mean(dist2nosehip)

        gt_lefthand = kpts[:,91:112,:]
        leftVolar = kpts[:,91,:].unsqueeze(dim=1)
        exceptLeftvolar = gt_lefthand - leftVolar
        dist2leftvolar = torch.sqrt(torch.square(exceptLeftvolar[:,:,0])+torch.square(exceptLeftvolar[:,:,1]))
        distLefthand = torch.mean(dist2leftvolar)

        gt_righthand = kpts[:,112:133,:]
        rightVolar = kpts[:,112,:].unsqueeze(dim=1)
        exceptRightvolar = gt_righthand - rightVolar
        dist2rightvolar = torch.sqrt(torch.square(exceptRightvolar[:,:,0])+torch.square(exceptRightvolar[:,:,1]))
        distRighthand = torch.mean(dist2rightvolar)

        # gt_dist = torch.sum(distBody, distFace, distLefthand, distRighthand)

        reverse_trans = [t["reverse_trans"] for t in targets]
        orig = transforms.get_final_preds(logits, reverse_trans, post_processing=True)
        orig_out = torch.tensor(orig[0]).to(device)
        pred_keypoints = orig_out[:,0:23, :]
        pred_nose = pred_keypoints[:, 0, :].unsqueeze(dim=1)
        pred_exceptNose = pred_keypoints - pred_nose
        pred_dist2nose = torch.sqrt(torch.square(pred_exceptNose[:, :, 0]) + torch.square(pred_exceptNose[:, :, 1]))
        pred_leftHip = pred_keypoints[:, 11, :].unsqueeze(dim=1)
        pred_exceptLefthip = pred_keypoints - pred_leftHip
        pred_dist2lefthip = torch.sqrt(torch.square(pred_exceptLefthip[:, :, 0]) + torch.square(pred_exceptLefthip[:, :, 1]))
        pred_rightHip = pred_keypoints[:, 12, :].unsqueeze(dim=1)
        pred_exceptRighthip = pred_keypoints - pred_rightHip
        pred_dist2righthip = torch.sqrt(torch.square(pred_exceptRighthip[:, :, 0]) + torch.square(pred_exceptRighthip[:, :, 1]))
        pred_distBody = torch.mean(torch.cat((pred_dist2nose, pred_dist2lefthip, pred_dist2righthip)))


        # pred_foot = orig_out[:, 17:23, :]
        pred_face = orig_out[:, 23:91, :]
        pred_nosehip = orig_out[:, 53, :].unsqueeze(dim=1)
        pred_exceptNosehip = pred_face - pred_nosehip
        pred_dist2nosehip = torch.sqrt(torch.square(pred_exceptNosehip[:, :, 0]) + torch.square(pred_exceptNosehip[:, :, 1]))
        pred_distFace = torch.mean(pred_dist2nosehip)

        pred_lefthand = orig_out[:, 91:112, :]
        pred_leftVolar = orig_out[:, 91, :].unsqueeze(dim=1)
        pred_exceptLeftvolar = pred_lefthand - pred_leftVolar
        pred_dist2leftvolar = torch.sqrt(torch.square(pred_exceptLeftvolar[:, :, 0]) + torch.square(pred_exceptLeftvolar[:, :, 1]))
        pred_distLefthand = torch.mean(pred_dist2leftvolar)

        pred_righthand = orig_out[:, 112:133, :]
        pred_rightVolar = orig_out[:, 112, :].unsqueeze(dim=1)
        pred_exceptRightvolar = pred_righthand - pred_rightVolar
        pred_dist2rightvolar = torch.sqrt(torch.square(pred_exceptRightvolar[:, :, 0]) + torch.square(pred_exceptRightvolar[:, :, 1]))
        pred_distRighthand = torch.mean(pred_dist2rightvolar)

        # pred_dist = torch.sum(pred_distBody, pred_distFace, pred_distLefthand, pred_distRighthand)


        # [B, num_kps, H, W] -> [B, num_kps]
        loss_mse = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss_msesum = torch.sum(loss_mse * kps_weights) / bs
        loss_distbody = torch.abs(distBody - pred_distBody)
        loss_distface = torch.abs(distFace - pred_distFace)
        loss_distlefthand = torch.abs(distLefthand - pred_distLefthand)
        loss_distrighthand = torch.abs(distRighthand - pred_distRighthand)
        loss = loss_msesum  +loss_distbody + loss_distface + loss_distlefthand + loss_distrighthand
        return loss
