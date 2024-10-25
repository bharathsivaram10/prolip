""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
import os
import csv

import fire
from tqdm import tqdm

import numpy as np
import torch

from analysis import load_model
from PIL import Image


class HierarchyDataset(torch.utils.data.Dataset):
    def __init__(self, hierarcaps_csv_fname, image_dir, tokenizer, transform):
        self.captions = []
        self.image_paths = []
        with open(hierarcaps_csv_fname, 'r') as fin:
            rows = csv.reader(fin, delimiter=',', quotechar='"')
            for idx, row in tqdm(enumerate(rows)):
                if idx == 0:
                    # skip header
                    continue
                _id, _captions, _image_url = row
                _captions = [c.strip() for c in _captions.split('=>')]
                self.captions.append(_captions)
                iid = _image_url.split('/')[-1]
                self.image_paths.append(os.path.join(image_dir, iid))
        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(Image.open(self.image_paths[index]))
        texts = self.tokenizer(self.captions[index])

        return img, texts, index

    def __len__(self):
        return len(self.image_paths)


def hcaps_det_eval(image_features, text_features, root, captions, n_steps=50, root_only=False):
    N = len(image_features)
    precisions = []
    recalls = []
    recalls1 = []
    recalls2 = []
    recalls3 = []
    recalls4 = []
    any_recalls = []
    n_s_preds = []
    for iidx in (pbar := tqdm(range(N))):
        image_feature = image_features[iidx]
        _, ind = (text_features @ image_feature.T).reshape(-1).sort()
        start = text_features[ind[-1] // 4, ind[-1] - ind[-1] // 4 * 4]
        step = (start - root) / n_steps

        s_preds = set()
        qs = []
        if root_only:
            qs = [root]
            n_steps = 1
        else:
            for i in range(n_steps):
                qs.append((start - step * i).T / torch.norm(start - step * i))
        qs = torch.stack(qs)
        ranks = (text_features @ qs.T).reshape(4000, n_steps).argmax(dim=0)
        for _rank in ranks:
            item_id = _rank // 4
            caption_id = _rank - _rank // 4 * 4
            s_preds.add(captions[item_id.item()][caption_id.item()])
        s_gts = set(captions[iidx])

        tp = len(s_preds & s_gts)
        fp = len(s_preds - s_gts)

        precision = tp / max(1, tp + fp)
        recall = np.mean([captions[iidx][i] in s_preds for i in range(4)])
        precisions.append(precision)
        recalls.append(recall)
        any_recalls.append(int(tp > 0))
        recalls1.append(int(captions[iidx][0] in s_preds))
        recalls2.append(int(captions[iidx][1] in s_preds))
        recalls3.append(int(captions[iidx][2] in s_preds))
        recalls4.append(int(captions[iidx][3] in s_preds))
        n_s_preds.append(len(s_preds))
        pbar.set_description(f"[DET] {np.mean(precisions)=} {np.mean(recalls)=} {np.mean(recalls1)=} {np.mean(recalls2)=} {np.mean(recalls3)=} {np.mean(recalls4)=} {np.mean(any_recalls)=} {np.mean(n_s_preds)=}")
    return 100 * np.mean(precisions), 100 * np.mean(recalls), 100 * np.mean(any_recalls)


def hcaps_unc_eval(image_features, image_stds, text_features, text_stds, captions, n_steps=50, n_root=1, base_root=None, root_only=False):
    N = len(image_features)
    precisions = []
    recalls = []
    recalls1 = []
    recalls2 = []
    recalls3 = []
    recalls4 = []
    any_recalls = []
    n_s_preds = []
    for iidx in (pbar := tqdm(range(N))):
        image_feature = image_features[iidx]
        _, ind = (text_features @ image_feature.T).reshape(-1).sort()
        tidx1 = ind[-1] // 4
        tidx2 = ind[-1] - ind[-1] // 4 * 4
        start = text_features[tidx1, tidx2]

        included = inclusion_test_batchwise(
            image_features[iidx].unsqueeze(0), image_stds.float()[iidx].unsqueeze(0),
            text_features.reshape(N * 4, -1), text_stds.float().reshape(N * 4, -1)
        )
        _, indices = included.sort()
        root = []
        for i in range(n_root):
            root.append(text_features[indices[0][-1] // 4, indices[0][-1] - indices[0][-1] // 4 * 4])
        if base_root is not None:
            root.append(base_root)
        root = sum(root)
        root = root / torch.norm(root)

        step = (start - root) / n_steps

        s_preds = set()
        qs = []
        if root_only:
            qs = [root]
            n_steps = 1
        else:
            for i in range(n_steps):
                qs.append((start - step * i).T / torch.norm(start - step * i))
        qs = torch.stack(qs)
        sims = (text_features @ qs.T).reshape(4000, n_steps)
        ranks = sims.argmax(dim=0)
        for _rank in ranks:
            item_id = _rank // 4
            caption_id = _rank - _rank // 4 * 4
            s_preds.add(captions[item_id.item()][caption_id.item()])
        s_gts = set(captions[iidx])

        tp = len(s_preds & s_gts)
        fp = len(s_preds - s_gts)

        precision = tp / max(1, tp + fp)
        recall = np.mean([captions[iidx][i] in s_preds for i in range(4)])
        precisions.append(precision)
        recalls.append(recall)
        any_recalls.append(int(tp > 0))
        recalls1.append(int(captions[iidx][0] in s_preds))
        recalls2.append(int(captions[iidx][1] in s_preds))
        recalls3.append(int(captions[iidx][2] in s_preds))
        recalls4.append(int(captions[iidx][3] in s_preds))
        n_s_preds.append(len(s_preds))
        pbar.set_description(f"[UNC] {np.mean(precisions)=} {np.mean(recalls)=} {np.mean(recalls1)=} {np.mean(recalls2)=} {np.mean(recalls3)=} {np.mean(recalls4)=} {np.mean(any_recalls)=} {np.mean(n_s_preds)=}")
    return 100 * np.mean(precisions), 100 * np.mean(recalls), 100 * np.mean(any_recalls)


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


@torch.no_grad()
def main(checkpoint_dir, hcaps_csv_fname, coco_val_img_dir, with_uncertainty=False, with_uncertainty_with_root=False, n_root=1, root_only=False):
    loaded = load_model(checkpoint_dir, is_dp=True)
    args = loaded['args']
    model = loaded['model']
    tokenizer = loaded['tokenizer']
    autocast = loaded['autocast']
    input_dtype = loaded['input_dtype']
    preprocess_val = loaded["preprocess_val"]
    hdata = HierarchyDataset(hcaps_csv_fname, coco_val_img_dir, tokenizer, preprocess_val)
    dataloader = torch.utils.data.DataLoader(hdata, batch_size=32, num_workers=8)
    text_features = []
    text_stds = []
    image_features = []
    image_stds = []
    image_paths = []
    captions = []
    for img, texts, index in tqdm(dataloader):
        img = img.to(device=args.device, dtype=input_dtype)
        texts = texts.to(device=args.device, dtype=input_dtype)
        N = len(texts)
        with autocast():
            output = model(text=texts.reshape(N * 4, -1))["text_features"]
            text_features.append(output["mean"].reshape(N, 4, -1).cpu())
            if with_uncertainty or with_uncertainty_with_root:
                text_stds.append(output["std"].reshape(N, 4, -1).cpu())

            output = model(image=img)["image_features"]
            image_features.append(output["mean"].cpu())
            if with_uncertainty or with_uncertainty_with_root:
                image_stds.append(output["std"].cpu())
        image_paths.extend([hdata.image_paths[idx] for idx in index])
        captions.extend([hdata.captions[idx] for idx in index])
    text_features = torch.cat(text_features)
    image_features = torch.cat(image_features)
    if with_uncertainty or with_uncertainty_with_root:
        text_stds = torch.cat(text_stds)
        image_stds = torch.cat(image_stds)

    if not with_uncertainty and not with_uncertainty_with_root:
        root_text_feature = model(text=tokenizer([""]).to(device=args.device, dtype=input_dtype))["text_features"]["mean"][0]
        hcaps_det_eval(image_features, text_features, root_text_feature.cpu(), captions, root_only=root_only)
    elif with_uncertainty_with_root:
        root_text_feature = model(text=tokenizer([""]).to(device=args.device, dtype=input_dtype))["text_features"]["mean"][0]
        hcaps_unc_eval(image_features, image_stds, text_features, text_stds, captions, n_root=n_root, base_root=root_text_feature.cpu(), root_only=root_only)
    else:
        hcaps_unc_eval(image_features, image_stds, text_features, text_stds, captions, n_root=n_root, root_only=root_only)


if __name__ == '__main__':
    fire.Fire(main)
