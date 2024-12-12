# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# __author__ = "Difei Tang"
# __email__ = "DIT18@pitt.edu"

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('mtl_fixmatch')
class MTLFixMatch(AlgorithmBase):

    """
        Adapted Mult-tasking Learning Version
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb_w):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']

            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            rows, cols = y_lb.size()
            sup_loss = torch.tensor(0.0).cuda()
            ws = [1, 0.5]
            for c in range(cols):
                logits_lb = logits_x_lb[c].view(-1, self.num_classes[c])
                labels_lb = y_lb[:, c].view(-1)

                sup_loss += self.ce_loss(logits_lb, labels_lb, reduction='mean') * ws[c]

            # compute unsup loss for context task
            logits_x_ulb_w0 = logits_x_ulb_w[0].detach()

            # get index of ID unlabeled data
            ulb_labels = y_ulb_w[:, 0].view(-1)
            ulb_idx = (ulb_labels == -200)

            logits_ulb_w0 = logits_x_ulb_w0[ulb_idx, :]

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_ulb_w0)

            # if distribution alignment hook is registered, call it
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # strong augmented logits
            logits_x_ulb_s_ = logits_x_ulb_s[0]
            logits_ulb_s = logits_x_ulb_s_[ulb_idx, :]

            ul_unsup_loss = self.consistency_loss(logits_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            #sup loss in unsup samples for relation extraction
            y_ulb_w_id = y_ulb_w[ulb_idx].long()
            labels_ulb1 = y_ulb_w_id[:, 1].view(-1).long()

            logits_x_ulb_w1 = logits_x_ulb_w[1].detach()
            logits_ulb_w1 = logits_x_ulb_w1[ulb_idx, :]

            ul_sup_loss = self.ce_loss(logits_ulb_w1, labels_ulb1, reduction='mean')

            # FIXME:
            unsup_loss = ul_unsup_loss

            # compute total loss
            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]