from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy import double
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import DeepBioisostere


class Trainer(nn.Module):
    r"""
    Model trainer with negative sampling method.
    """

    def __init__(
        self,
        model: DeepBioisostere,
        optimizer: Optimizer,
        num_neg_sample: int,
        batch_size: int,
        device: torch.device,
        profiler: torch.profiler,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.num_neg_sample = num_neg_sample
        self.batch_size = batch_size
        self.device = device
        self.profiler = profiler

    def set_data_loaders(self, train_dl, val_dl):
        self.train_dl = train_dl
        self.val_dl = val_dl

    def train(self) -> Tuple[double, double, double]:
        r"""
        About batch_data.
          x_n                (bool) : [N,F_n], for node (atom)
          edge_index_n       (int)  : [2,E_n], for node (atom) and model
          edge_attr_n        (bool) : [E_n,F_e], for node (atom)
          x_f                (int)  : [N], node to frag indice matcher
          edge_index_f       (int)  : [2,E_f], frag-level connectivity information
          edge_attr_f        (int)  : [2,E_f], atom indice for each frag-level edges

          smiles             (str)  : [B], SMILES of original molecule
          y_fragID           (int)  : [B], Fragment ID for which will be inserted (answer frag ID)
          y_pos_subgraph     (bool) : [pos_F], positive subgraphs for original molecule
          y_pos_subgraph_idx (bool) : [pos_F], positive subgraphs scatter indices for original molecule
          y_neg_subgraph     (bool) : [neg_F], allowed but negative subgraphs for original molecule
          y_neg_subgraph_idx (bool) : [neg_F], allowed but negative subgraphs scatter indices for original molecule
        """
        ppos_loss_list = []
        npos_loss_list = []
        pf_loss_list = []
        nf_loss_list = []
        att_loss_list = []

        ppos_prob_list = []
        npos_prob_list = []
        pf_prob_list = []
        nf_prob_list = []
        if self.profiler:
            self.profiler.start()
        for i_batch, batch in enumerate(self.train_dl):
            if self.profiler:
                if i_batch >= (2 + 2 + 5) * 2:
                    print("Break")
                    break

            batch["data"].to(self.device)
            batch["pos"].to(self.device)
            batch["neg"].to(self.device)
            for prop in self.model.properties:
                batch[prop] = batch[prop].to(self.device)

            self.optimizer.zero_grad()
            (
                pPosLoss,
                nPosLoss,
                pFragsLoss,
                nFragsLoss,
                attLoss,
                pPosProb,
                nPosProb,
                pFragsProb,
                nFragsProb,
                # ) = self.model(batch_data, batch_pos_frags, batch_neg_frags)
            ) = self.model(batch)
            # loss_to_update = (
            #    pPosLoss + nPosLoss + pFragsLoss + nFragsLoss * self.num_neg_sample
            # )
            loss_to_update = pPosLoss + nPosLoss + pFragsLoss + nFragsLoss + attLoss
            loss_to_update.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            ppos_loss_list.append(pPosLoss.detach().cpu().numpy())
            npos_loss_list.append(nPosLoss.detach().cpu().numpy())
            pf_loss_list.append(pFragsLoss.detach().cpu().numpy())
            nf_loss_list.append(nFragsLoss.detach().cpu().numpy())
            att_loss_list.append(attLoss.detach().cpu().numpy())
            ppos_prob_list.append(pPosProb.detach().cpu().numpy())
            npos_prob_list.append(nPosProb.detach().cpu().numpy())
            pf_prob_list.append(pFragsProb.detach().cpu().numpy())
            nf_prob_list.append(nFragsProb.detach().cpu().numpy())
            if self.profiler:
                self.profiler.step()

        if self.profiler:
            self.profiler.stop()
            exit()

        pposition_loss = np.mean(np.array(ppos_loss_list))
        nposition_loss = np.mean(np.array(npos_loss_list))
        pfrags_loss = np.mean(np.array(pf_loss_list))
        nfrags_loss = np.mean(np.array(nf_loss_list))
        att_loss = np.mean(np.array(att_loss_list))

        ppos_prob = np.mean(np.array(ppos_prob_list))
        npos_prob = np.mean(np.array(npos_prob_list))
        pfrags_prob = np.mean(np.array(pf_prob_list))
        nfrags_prob = np.mean(np.array(nf_prob_list))
        return (
            pposition_loss,
            nposition_loss,
            pfrags_loss,
            nfrags_loss,
            att_loss,
            ppos_prob,
            npos_prob,
            pfrags_prob,
            nfrags_prob,
        )

    @torch.no_grad()
    def validate(self) -> Tuple[double, double, double]:
        r"""
        About batch_data.
          x: [N,F_n]
          edge_index: [2,E]
          edge_attr: [E,F_e]
          smiles: [B]
          batch:[N]
        """
        ppos_loss_list = []
        npos_loss_list = []
        pf_loss_list = []
        nf_loss_list = []
        att_loss_list = []

        ppos_prob_list = []
        npos_prob_list = []
        pf_prob_list = []
        nf_prob_list = []
        for i_batch, batch in enumerate(self.val_dl):
            batch["data"].to(self.device)
            batch["pos"].to(self.device)
            batch["neg"].to(self.device)
            for prop in self.model.properties:
                batch[prop] = batch[prop].to(self.device)

            (
                pPosLoss,
                nPosLoss,
                pFragsLoss,
                nFragsLoss,
                attLoss,
                pPosProb,
                nPosProb,
                pFragsProb,
                nFragsProb,
                # ) = self.model(batch_data, batch_pos_frags, batch_neg_frags)
            ) = self.model(batch)
            ppos_loss_list.append(pPosLoss.detach().cpu().numpy())
            npos_loss_list.append(nPosLoss.detach().cpu().numpy())
            pf_loss_list.append(pFragsLoss.detach().cpu().numpy())
            nf_loss_list.append(nFragsLoss.detach().cpu().numpy())
            att_loss_list.append(attLoss.detach().cpu().numpy())
            ppos_prob_list.append(pPosProb.detach().cpu().numpy())
            npos_prob_list.append(nPosProb.detach().cpu().numpy())
            pf_prob_list.append(pFragsProb.detach().cpu().numpy())
            nf_prob_list.append(nFragsProb.detach().cpu().numpy())

        pposition_loss = np.mean(np.array(ppos_loss_list))
        nposition_loss = np.mean(np.array(npos_loss_list))
        pfrags_loss = np.mean(np.array(pf_loss_list))
        nfrags_loss = np.mean(np.array(nf_loss_list))
        att_loss = np.mean(np.array(att_loss_list))

        ppos_prob = np.mean(np.array(ppos_prob_list))
        npos_prob = np.mean(np.array(npos_prob_list))
        pfrags_prob = np.mean(np.array(pf_prob_list))
        nfrags_prob = np.mean(np.array(nf_prob_list))
        return (
            pposition_loss,
            nposition_loss,
            pfrags_loss,
            nfrags_loss,
            att_loss,
            ppos_prob,
            npos_prob,
            pfrags_prob,
            nfrags_prob,
        )

    @torch.no_grad()
    def test(self, test_dl) -> Tuple[double, double, double]:
        """
        About batch_data.
          x: [N,F_n]
          edge_index: [2,E]
          edge_attr: [E,F_e]
          smiles: [B]
          batch:[N]
        """
        ppos_loss_list = []
        npos_loss_list = []
        pf_loss_list = []
        nf_loss_list = []
        att_loss_list = []

        ppos_prob_list = []
        npos_prob_list = []
        pf_prob_list = []
        nf_prob_list = []
        for i_batch, batch in enumerate(self.test_dl):
            batch["data"].to(self.device)
            batch["pos"].to(self.device)
            batch["neg"].to(self.device)
            for prop in self.model.properties:
                batch[prop] = batch[prop].to(self.device)

            (
                pPosLoss,
                nPosLoss,
                pFragsLoss,
                nFragsLoss,
                attLoss,
                pPosProb,
                nPosProb,
                pFragsProb,
                nFragsProb,
                # ) = self.model(batch_data, batch_pos_frags, batch_neg_frags)
            ) = self.model(batch)
            ppos_loss_list.append(pPosLoss.detach().cpu().numpy())
            npos_loss_list.append(nPosLoss.detach().cpu().numpy())
            pf_loss_list.append(pFragsLoss.detach().cpu().numpy())
            nf_loss_list.append(nFragsLoss.detach().cpu().numpy())
            att_loss_list.append(attLoss.detach().cpu().numpy())
            ppos_prob_list.append(pPosProb.detach().cpu().numpy())
            npos_prob_list.append(nPosProb.detach().cpu().numpy())
            pf_prob_list.append(pFragsProb.detach().cpu().numpy())
            nf_prob_list.append(nFragsProb.detach().cpu().numpy())

        pposition_loss = np.mean(np.array(ppos_loss_list))
        nposition_loss = np.mean(np.array(npos_loss_list))
        pfrags_loss = np.mean(np.array(pf_loss_list))
        nfrags_loss = np.mean(np.array(nf_loss_list))
        att_loss = np.mean(np.array(att_loss_list))

        ppos_prob = np.mean(np.array(ppos_prob_list))
        npos_prob = np.mean(np.array(npos_prob_list))
        pfrags_prob = np.mean(np.array(pf_prob_list))
        nfrags_prob = np.mean(np.array(nf_prob_list))
        return (
            pposition_loss,
            nposition_loss,
            pfrags_loss,
            nfrags_loss,
            att_loss,
            ppos_prob,
            npos_prob,
            pfrags_prob,
            nfrags_prob,
        )

    # @torch.no_grad()
    # def save_frags_h(self):
    #    """
    #    self.model.frag_h: [frag_lib_size, F]
    #    """
    #    self.model._save_frags_h(self.val_frags_lib_dl)

    @torch.no_grad()
    def remove_frags_h(self):
        self.model._remove_frags_h()


class LR_Scheduler(ReduceLROnPlateau):
    def step(self, metrics) -> bool:
        """
        Return : whether to finish the training process or not.
        """
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            if not self._reduce_lr(epoch):
                return True
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        return False

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    print(
                        "Epoch {}: reducing learning rate"
                        " of group {} to {:.4e}.".format(epoch_str, i, new_lr)
                    )
            else:
                return False
        else:
            return True
