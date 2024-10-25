""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import os
import json

import fire
from tqdm import tqdm

import numpy as np
import torch

from analysis import load_model
from training.data import get_data
from prolip import build_zero_shot_classifier_with_stds, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


A_PHOTO_OF_TEMPLATES = ["A photo of {c}"]


def by_std_values_per_class(classifier_all, text_stds_all, coeff, **kwargs):
    classifier = []
    text_stds = []
    for cls_idx in range(classifier_all.size()[0]):
        unc_scores = text_stds_all[cls_idx].sum(-1)
        threshold = torch.mean(unc_scores) + coeff * torch.std(unc_scores)

        filtered_indices = torch.where(unc_scores <= threshold)[0]
        if len(filtered_indices) < 1:
            filtered_indices = torch.where(unc_scores == unc_scores.min())[0]
        print(len(filtered_indices), end=' ')
        clsf = classifier_all[cls_idx][filtered_indices].mean(0)
        classifier.append(clsf / clsf.norm())
        text_stds.append(text_stds_all[cls_idx][filtered_indices].mean(0))
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def by_std_values_all(classifier_all, text_stds_all, coeff, **kwargs):
    classifier = []
    text_stds = []
    n_filters = coeff
    threshold = text_stds_all.sum(-1).reshape(-1).sort()[0][:n_filters][-1]
    for cls_idx in range(classifier_all.size()[0]):
        unc_scores = text_stds_all[cls_idx].sum(-1)
        filtered_indices = torch.where(unc_scores <= threshold)[0]
        if len(filtered_indices) < 1:
            filtered_indices = torch.where(unc_scores == unc_scores.min())[0]
        print(len(filtered_indices), end=' ')
        clsf = classifier_all[cls_idx][filtered_indices].mean(0)
        classifier.append(clsf / clsf.norm())
        text_stds.append(text_stds_all[cls_idx][filtered_indices].mean(0))
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def by_ranking(classifier_all, text_stds_all, coeff, **kwargs):
    classifier = []
    text_stds = []
    for cls_idx in range(classifier_all.size()[0]):
        unc_scores = text_stds_all[cls_idx].sum(-1)
        threshold = unc_scores.sort()[0][:coeff][-1]

        filtered_indices = torch.where(unc_scores <= threshold)[0]
        if len(filtered_indices) < 1:
            filtered_indices = torch.where(unc_scores == unc_scores.min())[0]
        print(len(filtered_indices), end=' ')
        clsf = classifier_all[cls_idx][filtered_indices].mean(0)
        classifier.append(clsf / clsf.norm())
        text_stds.append(text_stds_all[cls_idx][filtered_indices].mean(0))
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def by_ranking_classwise(classifier_all, text_stds_all, coeff, **kwargs):
    classifier = []
    text_stds = []
    for cls_idx in range(classifier_all.size()[0]):
        unc_scores = text_stds_all[cls_idx].sum(-1)
        threshold = unc_scores.sort()[0][:coeff[cls_idx]][-1]

        filtered_indices = torch.where(unc_scores <= threshold)[0]
        if len(filtered_indices) < 1:
            filtered_indices = torch.where(unc_scores == unc_scores.min())[0]
        print(len(filtered_indices), end=' ')
        clsf = classifier_all[cls_idx][filtered_indices].mean(0)
        classifier.append(clsf / clsf.norm())
        text_stds.append(text_stds_all[cls_idx][filtered_indices].mean(0))
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def by_abs_value(classifier_all, text_stds_all, coeff, **kwargs):
    classifier = []
    text_stds = []
    for cls_idx in range(classifier_all.size()[0]):
        unc_scores = text_stds_all[cls_idx].sum(-1)
        threshold = coeff

        filtered_indices = torch.where(unc_scores <= threshold)[0]
        if len(filtered_indices) < 1:
            filtered_indices = torch.where(unc_scores == unc_scores.min())[0]
        print(len(filtered_indices), end=' ')
        clsf = classifier_all[cls_idx][filtered_indices].mean(0)
        classifier.append(clsf / clsf.norm())
        text_stds.append(text_stds_all[cls_idx][filtered_indices].mean(0))
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def estimate_mixing_proportions(observations, means, vars, tol=1e-12, max_iter=1000, alpha=100, var_multi=1, stable_eps=None, prior=None, device='cpu'):
    """
    Estimates the mixing proportions for known Gaussian models using the EM algorithm.

    Parameters:
    - observations: torch.Tensor of shape (N, D), the data points.
    - models: list of dictionaries, each with 'mean' and 'variance' tensors of shape (D,).
    - tol: float, convergence tolerance.
    - max_iter: int, maximum number of iterations.

    Returns:
    - pi: torch.Tensor of shape (K,), estimated mixing proportions.
    """
    K = len(means)
    N, D = observations.shape

    if prior is None or prior == "uniform":
        # Initialize mixing proportions uniformly
        pi = torch.ones(K) / K
    elif prior == "inverse":
        pi = 1 / vars.mean(dim=1)
        pi = pi / pi.sum()
    elif prior == "inverse_sq":
        pi = 1 / (vars ** 2).mean(dim=1)
        pi = pi / pi.sum()
    elif prior == "inverse_5":
        pi = 1 / (vars ** 5).mean(dim=1)
        pi = pi / pi.sum()

    # Set Dirichlet prior concentration parameters
    alpha = torch.ones(K) * alpha

    log_two_pi = np.log(2 * np.pi)

    observations = observations.to(device)
    means = means.to(device)
    vars = vars.to(device) * var_multi
    if stable_eps:
        new_vars = vars + stable_eps * torch.ones_like(vars)
        print(f"{vars.mean(dim=1).max()=} {vars.mean(dim=1).min()=} {new_vars.mean(dim=1).max()=} {new_vars.mean(dim=1).min()=} ", end="\r")
        vars = new_vars
    pi = pi.to(device)
    alpha = alpha.to(device)

    observations = observations.unsqueeze(1)     # Shape: (N, 1, D)
    means = means.unsqueeze(0)              # Shape: (1, K, D)
    vars = vars.unsqueeze(0) * var_multi  # Shape: (1, K, D)

    # EM algorithm
    for iteration in range(max_iter):
        # E-Step: Compute responsibilities

        # Compute the exponent term
        diff = observations - means           # Shape: (N, K, D)
        exponent = -0.5 * (diff ** 2 / vars).sum(dim=2)  # Shape: (N, K)

        # Compute the log determinant term
        log_det = -0.5 * torch.sum(torch.log(vars), dim=2) - 0.5 * D * log_two_pi  # Shape: (1, K)

        # Compute log probabilities
        log_prob = log_det + exponent                # Shape: (N, K)

        # Add log mixing proportions
        log_pi = torch.log(pi)                       # Shape: (K,)
        log_responsibilities = log_prob + log_pi     # Shape: (N, K)

        # Log-sum-exp for normalization
        max_log_responsibility, _ = torch.max(log_responsibilities, dim=1, keepdim=True)
        log_responsibilities_shifted = log_responsibilities - max_log_responsibility
        responsibilities = torch.exp(log_responsibilities_shifted)
        sum_responsibilities = responsibilities.sum(dim=1, keepdim=True)
        responsibilities = responsibilities / sum_responsibilities

        # M-Step: Update mixing proportions with Dirichlet prior
        Nk = responsibilities.sum(dim=0)  # Shape: (K,)

        pi_new_numerator = Nk + alpha - 1
        pi_new_denominator = N + torch.sum(alpha - 1)
        pi_new = pi_new_numerator / pi_new_denominator

        # Ensure pi_new sums to 1
        pi_new = pi_new.clamp(min=0)
        pi_new = pi_new / pi_new.sum()

        # Check for convergence
        if torch.norm(pi_new - pi) < tol:
            break

        pi = pi_new

    return pi


@torch.no_grad()
def bprw(classifier_all, text_stds_all, coeff,
         dataloader, loaded, checkpoint_dir=None,
         em_weight_out_fname=None,
         use_gt=None,
         **kwargs):
    args = loaded['args']
    model = loaded['model']
    tokenizer = loaded['tokenizer']
    autocast = loaded['autocast']
    input_dtype = loaded['input_dtype']
    preprocess_train = loaded['preprocess_train']
    preprocess_val = loaded['preprocess_val']

    tkns = coeff.split('-')
    if len(tkns) == 6:
        alpha, multiplier, NN, K, stable_eps, prior = float(tkns[0]), float(tkns[1]), int(tkns[2]), int(tkns[3]), float(tkns[4]), tkns[5]
    else:
        alpha, multiplier, NN, K, stable_eps, prior = 100, 1, 100, 10, 0, "uniform"
    print(f"{coeff=} {alpha=} {multiplier=} {NN=} {K=} {stable_eps=} {prior=}")

    skip = False
    if checkpoint_dir:
        cache_fname = os.path.join(checkpoint_dir, "em_cache.pt")
        '''
        if not use_gt:
            cache_fname = os.path.join(checkpoint_dir, "em_cache.pt")
        else:
            cache_fname = os.path.join(checkpoint_dir, "em_cache_gt.pt")
        '''
        if os.path.exists(cache_fname):
            observations, stds = torch.load(cache_fname, map_location="cpu")
            skip = True
        else:
            with open(cache_fname, 'w'):
                # check
                pass

    if not skip:
        if use_gt:
            args.batch_size = 10
        data = get_data(args, (preprocess_train, preprocess_val), tokenizer=tokenizer)
        dataloader = data["imagenet-val"].dataloader

        observations = []
        stds = []
        for images, _ in tqdm(dataloader):
            images = images.to(device=args.device, dtype=input_dtype)
            with autocast():
                if not use_gt:
                    output = model(image=images)["image_features"]
                else:
                    output = model(image=images[0].unsqueeze(0))["image_features"]
                observations.append(output["mean"].detach().cpu())
                stds.append(output["std"].detach().cpu())
        observations = torch.cat(observations, dim=0)
        stds = torch.cat(stds, dim=0)
        print(f"{observations.shape=}")

    if checkpoint_dir:
        torch.save([observations, stds], cache_fname)

    classifier = []
    text_stds = []
    observations_cuda = observations.to(classifier_all.device)
    stds = stds.to(classifier_all.device)
    pis = []
    for cls_idx in (pbar := tqdm(range(classifier_all.size()[0]))):
        # observations: (N, D)
        # means: (K, D)
        if not use_gt:
            sims = classifier_all[cls_idx].mean(dim=0).to(observations.device) @ observations.T
            _, indices = torch.topk(sims, k=NN)
        else:
            if NN > 50:
                # XXX prevent overflow
                NN = 50
            indices = torch.LongTensor(list(range(50 * cls_idx, 50 * (cls_idx) + NN))).to(observations.device)

        mu = observations_cuda[indices]
        logsigma = stds[indices] / 2
        eps = torch.randn(mu.size(0), K, mu.size(1), dtype=mu.dtype, device=mu.device)
        samples = eps * logsigma.float().exp().unsqueeze(1) + mu.unsqueeze(1)
        samples = samples.reshape(-1, mu.size(1))

        w = estimate_mixing_proportions(
            samples, classifier_all[cls_idx].to(samples.device), text_stds_all[cls_idx].to(samples.device),
            alpha=alpha, var_multi=multiplier, stable_eps=stable_eps, prior=prior)
        pis.append(w)
        w = w.to(device=args.device)
        classifier.append(torch.matmul(w, classifier_all[cls_idx]))
        # text_stds.append(torch.matmul(w, text_stds_all[cls_idx]) * len(w))
        # text_stds.append(torch.matmul(w, text_stds_all[cls_idx]))
        text_stds.append(torch.matmul(w ** 2, text_stds_all[cls_idx]))
        values, indices = w.sort()
        pbar.set_description(f"[{cls_idx:03d}] {values[-3:]=} {indices[-3:]=}")
    if em_weight_out_fname:
        torch.save(pis, em_weight_out_fname)
    classifier = torch.stack(classifier).T
    text_stds = torch.stack(text_stds).T
    return classifier, text_stds


def inclusion_test_batchwise(mu1, logsigma_sq1, mu2, logsigma_sq2, eps=0):
    """ Test if mu1, logsigma_sq1 is included in mu2, logsigma_sq2
    The test returns a large value if 1 is included in 2, otherwise returns a small value
    """
    # Compute inverse variances
    inv_sigma_sq1 = torch.exp(-logsigma_sq1 + eps)  # Shape: (N, D)
    inv_sigma_sq2 = torch.exp(-logsigma_sq2 + eps)  # Shape: (M, D)

    # Expand dimensions to enable broadcasting
    inv_sigma_sq1_exp = inv_sigma_sq1[:, None, :]    # Shape: (N, 1, D)
    inv_sigma_sq2_exp = inv_sigma_sq2[None, :, :]    # Shape: (1, M, D)

    mu1_exp = mu1[:, None, :]                        # Shape: (N, 1, D)
    mu2_exp = mu2[None, :, :]                        # Shape: (1, M, D)
    logsigma_sq1_exp = logsigma_sq1[:, None, :]      # Shape: (N, 1, D)
    logsigma_sq2_exp = logsigma_sq2[None, :, :]      # Shape: (1, M, D)

    # Compute 'a', 'b', 'c' using broadcasting
    a = inv_sigma_sq1_exp + 0.5 * inv_sigma_sq2_exp
    b = 2 * mu1_exp * inv_sigma_sq1_exp + mu2_exp * inv_sigma_sq2_exp
    c = mu1_exp ** 2 * inv_sigma_sq1_exp + 0.5 * mu2_exp ** 2 * inv_sigma_sq2_exp

    # Compute the final result
    result = (
        -2 * logsigma_sq1_exp
        - logsigma_sq2_exp
        - 0.5 * torch.log(a)
        + b ** 2 / (4 * a)
        - c
    )  # Shape: (N, M, D)

    # Sum over the feature dimension 'D' to get the final (N, M) tensor
    result = result.sum(dim=2)  # Shape: (N, M)

    return result


def run_zeroshot(checkpoint_dir, filtering_method, imagenet_val_dir, coeffs=None, out_fname=None, batch_size=32, template="openai",
                 skip_classification=False, em_weight_out_fname=None, use_gt=False, classwise=False):
    """
    Table 3 hyperparameter:
        (Few-shot) bprw,2-1-5-20-0.02-inverse --use_gt
        (Use NN)   bprw,5-1-10-10-0.02-inverse
    """
    filtering_method_str = filtering_method
    if template == 'openai':
        n_templates = 80
    elif template == 'single':
        n_templates = 1

    if filtering_method == 'by_std_values_per_class':
        filtering_method = by_std_values_per_class
        assert coeffs
    elif filtering_method == 'by_std_values_all':
        filtering_method = by_std_values_all
        assert coeffs
    elif filtering_method == 'by_ranking':
        filtering_method = by_ranking
        assert coeffs
    elif filtering_method == 'by_abs_value':
        filtering_method = by_abs_value
        assert coeffs
    elif filtering_method == 'bprw':
        filtering_method = bprw
        assert coeffs
    elif filtering_method == 'greedy':
        filtering_method = by_ranking
        coeffs = list(range(1, n_templates + 1))
        classwise = True
    elif filtering_method == 'default':
        filtering_method = by_ranking
        coeffs = [n_templates]
    else:
        raise ValueError

    if not out_fname:
        out_fname = os.path.join(checkpoint_dir, f"{filtering_method_str}.txt")
    elif out_fname == 'skip':
        out_fname = None

    if isinstance(coeffs, list):
        pass
    else:
        coeffs = [coeffs]

    if out_fname:
        if os.path.exists(out_fname):
            coeffs_str = [str(c) for c in coeffs]
            with open(out_fname, 'r') as fin:
                prev_coeffs = set([row.split(',')[1] for row in fin])

            coeffs = [coeff for idx, coeff in enumerate(coeffs) if coeffs_str[idx] not in prev_coeffs]
            if len(coeffs_str) != len(coeffs):
                print('Found existing coefficients. Skipping the existing coeffs.')
        else:
            with open(out_fname, 'w'):
                # initialize
                pass

    loaded = load_model(checkpoint_dir, is_dp=False, batch_size=batch_size)
    args = loaded['args']
    model = loaded['model']
    tokenizer = loaded['tokenizer']
    autocast = loaded['autocast']
    input_dtype = loaded['input_dtype']
    preprocess_train = loaded['preprocess_train']
    preprocess_val = loaded['preprocess_val']

    args.imagenet_val = imagenet_val_dir
    args.coco_test = None
    data = get_data(args, (preprocess_train, preprocess_val), tokenizer=tokenizer)

    if template == 'openai':
        templates = OPENAI_IMAGENET_TEMPLATES
    elif template == 'single':
        templates = A_PHOTO_OF_TEMPLATES

    with autocast():
        classifier_all, text_stds_all = build_zero_shot_classifier_with_stds(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
            no_aggregate=True,
        )

    print(f"{coeffs=} Start!")
    for coeff in coeffs:
        print(f"{coeff=}")
        dataloader = data["imagenet-val"].dataloader
        classifier, text_stds = filtering_method(
            classifier_all, text_stds_all, coeff,
            dataloader=dataloader, loaded=loaded, checkpoint_dir=checkpoint_dir,
            em_weight_out_fname=em_weight_out_fname,
            use_gt=use_gt,
        )

        args.batch_size = batch_size
        all_correct, all_unc_correct, all_n_items = 0, 0, 0
        std_bins = {idx: [] for idx in range(1000)}
        all_std_bins = []
        corrects = {idx: [] for idx in range(1000)}
        unc_corrects = {idx: [] for idx in range(1000)}
        unc_weight = text_stds.sum(dim=0)

        avg_var_log = text_stds.log().mean()
        print(f"{avg_var_log=}")

        if skip_classification:
            print("skip classification...")
            continue

        def accuracy(output, target, topk=(1,)):
            pred = output.topk(max(topk), 1, True, True)[1].t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk], correct

        with torch.no_grad():
            for images, target in (pbar := tqdm(dataloader, unit_scale=args.batch_size)):
                images = images.to(device=args.device, dtype=input_dtype)
                target = target.to(args.device)
                with autocast():
                    output = model(image=images)["image_features"]
                    image_features = output["mean"]
                    image_stds = output["std"]
                    logits = 100.0 * image_features @ classifier
                    _all_correct, _corrects = accuracy(logits, target, topk=(1,))
                    unc_logits = logits - 50.0 * unc_weight
                    _all_unc_correct, _unc_corrects = accuracy(unc_logits, target, topk=(1,))

                    # Update the corrects and std_bins for each class
                    for idx, label in enumerate(target):
                        std_bins[label.item()].append(torch.exp(image_stds[idx]).sum(dim=-1).item())
                        corrects[label.item()].append(_corrects[0][idx].item())
                        unc_corrects[label.item()].append(_unc_corrects[0][idx].item())
                    all_std_bins.extend(torch.exp(image_stds).mean(dim=-1).tolist())

                all_correct += sum(_all_correct)
                all_unc_correct += sum(_all_unc_correct)
                all_n_items += len(target)

                # Calculate min, max, mean, and median for overall std_bins (flattened)
                all_std_values = [std for sublist in std_bins.values() for std in sublist]
                pbar.set_description(f"Processing {all_n_items}, "
                                     f"unc acc={100*all_unc_correct/all_n_items:.2f} "
                                     f"acc={100*all_correct/all_n_items:.2f} "
                                     f"({np.min(all_std_values):.4f}, {np.max(all_std_values):.4f}, "
                                     f"{np.mean(all_std_values):.4f}, {np.median(all_std_values):.4f})")
        if out_fname:
            if use_gt:
                _out_fname = out_fname + '.gt'
            else:
                _out_fname = out_fname
            with open(_out_fname, 'a') as fout:
                if classwise:
                    out = {
                        "coeff": coeff,
                        "acc_per_class": [sum(corrects[idx]) / len(corrects[idx]) * 100 for idx in range(1000)],
                        "acc_per_class_unc": [sum(unc_corrects[idx]) / len(unc_corrects[idx]) * 100 for idx in range(1000)]
                    }
                    fout.write(json.dumps(out))
                    fout.write("\n")
                else:
                    fout.write(f"{filtering_method_str},{coeff},{100*all_correct/all_n_items},{100*all_unc_correct/all_n_items}\n")
        else:
            print("acc_per_class", [sum(corrects[idx]) / len(corrects[idx]) * 100 for idx in range(1000)])
            print("acc_per_class_unc", [sum(unc_corrects[idx]) / len(unc_corrects[idx]) * 100 for idx in range(1000)])


if __name__ == "__main__":
    fire.Fire(run_zeroshot)
