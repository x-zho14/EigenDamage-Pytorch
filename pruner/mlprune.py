import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from tqdm import tqdm
from utils.kfac_utils import ComputeCovA, ComputeCovG, fetch_mat_weights
from utils.common_utils import try_contiguous, PresetLRScheduler
from utils.network_utils import stablize_bn
from utils.prune_utils import get_threshold
import os
import matplotlib.pyplot as plt


class MLPruner:

    def __init__(self, model, config):
        self.known_modules = {'Linear', 'Conv2d'}
        # self.MatGradHandler = ComputeMatGrad()
        self.config = config
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
        print("=> We keep following layers in MLPruner. ")
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
            print(idx, m)
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
        # print(all_weights, norms)
        all_weights = sorted(all_weights)
        # print(all_weights)
        cuttoff_index = np.round(ratio * len(all_weights)).astype(int)
        cutoff = all_weights[cuttoff_index]
        print(cutoff)
        new_masks = dict()
        if prev_masks is None:
            prev_masks = dict()

        total = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                total += m.weight.data.numel()
        pruned = 0
        pruned2 = 0
        ratio_list = []
        idx = 0
        score_list = []
        score_layer_dict = {}
        for m in self.importances.keys():
            idx += 1
            imps = self.importances[m]
            mask = prev_masks.get(m, np.ones(m.weight.shape))
            print('Norm is: %s' % norms.get(m, 1))
            # import pdb; pdb.set_trace()
            scores_normalized = torch.flatten(imps/norms.get(m, 1.0))
            score_layer_dict[idx] = scores_normalized.tolist()
            plt.hist(scores_normalized.tolist(), bins=50, density=True)
            plt.xlim(0, scores_normalized.max().item())
            plt.xlabel("Scores")
            plt.ylabel("Frequecy")
            plt.title("Histogram of Scores of Layer "+str(idx)+" of MLPrune")
            plt.grid(True, linestyle="--")
            plt.savefig("resnet32" + "_" + str(idx) + "_" + str(0.001) + ".pdf", bbox_inches='tight')
            plt.clf()
            plt.cla()
            score_list.extend(scores_normalized.tolist())

            new_masks[m] = np.where(np.abs(imps.data.cpu().numpy()/norms.get(m, 1.0)) <= cutoff, np.zeros(mask.shape), mask)
            new_masks[m] = torch.from_numpy(new_masks[m]).float().cuda().requires_grad_(False)
            # weight_copy = m.weight.data.clone()

            # mask2 = self.importances[m].abs().gt(cutoff).float().cuda()
            # weight_copy.mul_(mask2)

            if int(torch.sum(new_masks[m])) == 0:
                print("pruned all")
                # temp_thre = get_threshold(np.abs(imps.view(-1).data.cpu().numpy()).tolist(), 0.999)
                # print("0.001 thre", temp_thre)
                # new_masks[m] = np.where(np.abs(imps.data.cpu().numpy() / norms.get(m, 1.0)) <= temp_thre,np.zeros(mask.shape), mask)
                # new_masks[m] = torch.from_numpy(new_masks[m]).float().cuda().requires_grad_(False)

            pruned = pruned + new_masks[m].numel() - torch.sum(new_masks[m])
            m.weight.data.mul_(new_masks[m])
            # pruned2 = pruned2 + new_masks[m].numel() - torch.sum(new_masks[m])
            # print("pruned pruned2", pruned, pruned2)
            # if weight_copy.equal(m.weight.data):
            #     print("correct")
            print('layer index: {:s} \t total params: {:d} \t remaining params: {:d} \t remaining ratio: {:f}'.
                format(str(m), new_masks[m].numel(), int(torch.sum(new_masks[m])), int(torch.sum(new_masks[m])) / new_masks[m].numel()))
            ratio_list.append(int(torch.sum(new_masks[m])) / new_masks[m].numel())
        print(ratio_list)
        n, bins, patches = plt.hist(score_list, bins=100, density=True)
        plt.xlim(0, max(score_list))
        plt.xlabel("Scores")
        plt.ylabel("Frequecy")
        plt.title("Histogram of Scores of Layer " + str(idx) + " of MLPrune")
        plt.grid(True, linestyle="--")
        plt.savefig("resnet32" + "_" + "whole"+ "_" + str(0.001) + ".pdf", bbox_inches='tight')

        import matplotlib
        fig = plt.figure(figsize=(9, 9))
        font = {'size': 11}
        matplotlib.rc('font', **font)
        plt.subplots_adjust(bottom=0.3)
        plt.subplot(141)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[1], bins=50)
        plt.xlim(0, 0.02)
        plt.xticks()
        plt.ylabel("# of Weights")
        plt.title("Layer " + str(1))

        plt.subplot(142)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[23], bins=50)
        plt.xlim(0, 0.0002)
        plt.title("Layer " + str(23))

        plt.subplot(143)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[26], bins=50)
        plt.xlim(0, 0.0001)
        plt.ylabel("# of Weights")
        plt.xlabel("Weight Importance Score")
        plt.title("Layer " + str(26))
        plt.savefig('histogram.pdf', bbox_inches='tight')

        plt.subplot(144)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[30], bins=50)
        plt.xlim(0, 0.0001)
        plt.xlabel("Weight Importance Score")
        plt.title("Layer " + str(30))
        plt.savefig('histogram_17.pdf', bbox_inches='tight')

        import matplotlib
        fig = plt.figure(figsize=(9, 9))
        font = {'size': 11}
        matplotlib.rc('font', **font)
        plt.subplots_adjust(bottom=0.3)
        plt.subplot(141)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[1], bins=50)
        plt.xlim(0, 0.02)
        plt.xticks()
        plt.ylabel("# of Weights")
        plt.title("Layer " + str(1))

        plt.subplot(142)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[23], bins=50)
        plt.xlim(0, 0.0002)
        plt.title("Layer " + str(23))

        plt.subplot(143)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[26], bins=50)
        plt.xlim(0, 0.0001)
        plt.ylabel("# of Weights")
        plt.xlabel("Weight Importance Score")
        plt.title("Layer " + str(26))
        # plt.savefig('histogram_18.pdf', bbox_inches='tight')

        plt.subplot(144)
        plt.grid(True, linestyle="--")
        ax = plt.gca()
        ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        plt.hist(score_layer_dict[30], bins=50)
        plt.xlim(0, 0.0001)
        plt.xlabel("Weight Importance Score")
        plt.title("Layer " + str(30))
        plt.savefig('histogram.pdf', bbox_inches='tight')
        #
        # for m in new_masks.keys():
        #
        #     m.weight.data.mul_(new_masks[m])

        print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))
        state = {
            'epoch': 0,
            'state_dict': self.model.state_dict(),
            'acc': 0,
            'best_acc': 0.,
        }
        torch.save(state, self.config.network +"_" +self.config.dataset +"_"+ str(ratio) +"_"+str(self.config.batch_size)+ 'pruned.pth.tar')
        return new_masks

    def compute_masks(self, dataloader, criterion, device, fisher_type, prune_ratio, normalize=False, prev_masks=None):
        print("1")
        self._prepare_model()
        print("2")
        self._compute_fisher(dataloader, criterion, device, fisher_type)
        print("3")
        self._update_inv()  # eigen decomposition of fisher
        print("4")
        self._get_unit_importance()
        print("5")
        new_masks = self._make_masks(prune_ratio, prev_masks, normalize)
        print("6")
        self._rm_hooks()
        print("7")
        self._clear_buffer()
        print("8")
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
                print(m)
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

    def fine_tune_model(self, trainloader, testloader, criterion, optim, learning_rate, weight_decay, nepochs=10,
                        device='cuda'):
        self.model = self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        # optimizer = optim.Adam(self.model.parameters(), weight_decay=5e-4)
        lr_schedule = {0: learning_rate, int(nepochs * 0.5): learning_rate * 0.1,
                       int(nepochs * 0.75): learning_rate * 0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)
        best_test_acc, best_test_loss = 0, 100
        iterations = 0
        for epoch in range(nepochs):
            self.model = self.model.train()
            correct = 0
            total = 0
            all_loss = 0
            lr_scheduler(optimizer, epoch)
            desc = ('[LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
            prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
            for batch_idx, (inputs, targets) in prog_bar:
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                iterations += 1
                all_loss += loss.item()
                loss.backward()
                for k, m in enumerate(self.model.modules()):
                    if isinstance(m, torch.nn.Conv2d):
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        m.weight.grad.data.mul_(mask)
                optimizer.step()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                desc = ('[%d][LR: %.5f, WD: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (epoch, lr_scheduler.get_lr(optimizer), weight_decay, all_loss / (batch_idx + 1),
                         100. * correct / total, correct, total))
                prog_bar.set_description(desc, refresh=True)
            test_loss, test_acc = self.test_model(testloader, criterion, device)
            best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
            best_test_acc = max(test_acc, best_test_acc)
        print('** Finetuning finished. Stabilizing batch norm and test again!')
        stablize_bn(self.model, trainloader)
        test_loss, test_acc = self.test_model(testloader, criterion, device)
        best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
        best_test_acc = max(test_acc, best_test_acc)
        return best_test_loss, best_test_acc

    def test_model(self, dataloader, criterion, device='cuda'):
        self.model = self.model.eval()
        correct = 0
        total = 0
        all_loss = 0
        desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (0, 0, correct, total))
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=True)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            #try:
            outputs = self.model(inputs)
            #except:
            #    import pdb; pdb.set_trace()
            loss = criterion(outputs, targets)
            all_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (all_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
        return all_loss / (batch_idx + 1), 100. * correct / total