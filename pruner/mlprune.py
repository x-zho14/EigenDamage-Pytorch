import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from tqdm import tqdm
from utils.kfac_utils import ComputeCovA, ComputeCovG, fetch_mat_weights
from utils.common_utils import try_contiguous


class MLPruner:

    def __init__(self, model):
        self.known_modules = {'Linear', 'Conv2d'}
        # self.MatGradHandler = ComputeMatGrad()
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.model = model
        self.modules = []
        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.importances = dict()

    def _save_input(self, module, input):
        aa = self.CovAHandler(input[0].data, module)
        # Initialize buffers
        if self.steps == 0:
            self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1.0))
        self.m_aa[module] += aa

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = self.CovGHandler(grad_output[0].data, module, True)
        # Initialize buffers
        if self.steps == 0:
            self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1.0))
        self.m_gg[module] += gg

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in OBDPruner. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _compute_fisher(self, dataloader, criterion, device='cuda', fisher_type='true'):
        self.mode = 'basis'
        self.model = self.model.eval()
        self.init_step()
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            if fisher_type == 'true':
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                              1).squeeze().to(device)
                # import pdb; pdb.set_trace()
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward()
            else:
                loss = criterion(outputs, targets)
                loss.backward()
            self.step()
        self.mode = 'quite'

    def init_step(self):
        self.steps = 0

    def _update_inv(self):
        assert self.steps > 0, 'At least one step before update inverse!'
        eps = 1e-20
        for idx, m in enumerate(self.modules):
            # m_aa, m_gg = normalize_factors(self.m_aa[m], self.m_gg[m])
            m_aa, m_gg = self.m_aa[m], self.m_gg[m]
            self.d_a[m], self.Q_a[m] = torch.symeig(m_aa + torch.diag(m_aa.new(m_aa.size(0)).fill_(1.0)), eigenvectors=True)
            self.d_g[m], self.Q_g[m] = torch.symeig(m_gg + torch.diag(m_gg.new(m_gg.size(0)).fill_(1.0)), eigenvectors=True)
            self.d_a[m].mul_((self.d_a[m] > eps).float())
            self.d_g[m].mul_((self.d_g[m] > eps).float())

        self._inversed = True

    def _fetch_weights_collections(self, _prev_masks, _normalize):
        weights = []
        eps = 0
        norms = dict()
        if _prev_masks is None:
            for m in self.importances.keys():
                imps = self.importances[m]
                w = np.abs(imps.view(-1).data.cpu().numpy())
                if _normalize:
                    norms[m] = (w.sum() + eps).item()
                    w = w / (w.sum() + eps)

                weights.extend(w.tolist())
        else:
            raise NotImplementedError
        return weights, norms

    def _make_masks(self, ratio, prev_masks, normalize):
        all_weights, norms = self._fetch_weights_collections(prev_masks, normalize)
        all_weights = sorted(all_weights)
        cuttoff_index = np.round(ratio * len(all_weights)).astype(int)
        cutoff = all_weights[cuttoff_index]
        new_masks = dict()
        if prev_masks is None:
            prev_masks = dict()
        for m in self.importances.keys():
            imps = self.importances[m]
            mask = prev_masks.get(m, np.ones(m.weight.shape))
            print('Norm is: %s' % norms.get(m, 1))
            # import pdb; pdb.set_trace()
            new_masks[m] = np.where(np.abs(imps.data.cpu().numpy()/norms.get(m, 1.0)) <= cutoff, np.zeros(mask.shape), mask)
        for m in new_masks.keys():
            new_masks[m] = torch.from_numpy(new_masks[m]).float().cuda().requires_grad_(False)
        return new_masks

    def compute_masks(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=False, prev_masks=None):
        self._prepare_model()
        self._compute_fisher(dataloader, criterion, device, fisher_type)
        self._update_inv()  # eigen decomposition of fisher

        self._get_unit_importance()
        new_masks = self._make_masks(prune_ratio, prev_masks, normalize)
        self._rm_hooks()
        self._clear_buffer()
        return new_masks

    def _rm_hooks(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname in self.known_modules:
                m._backward_hooks = OrderedDict()
                m._forward_pre_hooks = OrderedDict()

    def _get_unit_importance(self):
        eps = 1e-20
        assert self._inversed, 'Not inversed.'
        with torch.no_grad():
            for m in self.modules:
                w = fetch_mat_weights(m, False)  # output_dim * input_dim
                # (Q_a ⊗ Q_g) vec(W) = Q_g.t() @ W @ Q_a
                A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()
                A_inv_diag = torch.diag(A_inv)
                G_inv_diag = torch.diag(G_inv)
                w_imp = w ** 2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
                if isinstance(m, nn.Linear) and m.bias is not None:
                    self.importances[m] = try_contiguous(w_imp[:, :-1])
                elif isinstance(m, nn.Conv2d):
                    kh, kw = m.kernel_size
                    in_c = m.in_channels
                    out_c = m.out_channels
                    self.importances[m] = try_contiguous(w_imp.view(out_c, in_c, kh, kw))

    def step(self):
        self.steps += 1

    def _clear_buffer(self):
        self.Fisher = {}
        self.modules = []