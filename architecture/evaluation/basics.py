# Originally from https://github.com/xuebinqin/DIS
# Refactoring by SUNN(01139138@hyundai-autoever.com)

import torch
from typing import Dict

from . import EVALUATOR_REGISTRY
from utils.types import Dictconfigs


@EVALUATOR_REGISTRY.register()
class F1MaeTorch(object):
	def __init__(self, configs: Dictconfigs):
		# todo : criterion, optimizer 와 구조 통일을 위해 (사용하지 않는) configs 받아옴 > 수정 필요
		self.parameters = configs.parameters

	def __call__(self, pred, gt) -> Dict:
		if (len(gt.shape) > 2):
			gt = gt[:, :, 0]

		f1score_dict = self.f1score_torch(pred, gt)
		mae_dict = self.mae_torch(pred, gt)
		return f1score_dict.update(mae_dict)

	def mae(self, pred, gt) -> Dict:
		h, w = gt.shape[0:2]
		sum_error = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
		mae_error = torch.divide(sum_error, float(h) * float(w) * 255.0 + 1e-4)
		return dict(
			mae_error=mae_error.cpu().data.numpy()
		)

	def f1score(self, pred, gt) -> Dict:
		# todo : 128 의미 확인
		gtNum = torch.sum((gt > 128).float() * 1)  # number of ground truth pixels

		pp = pred[gt > 128]
		nn = pred[gt <= 128]
		pp_hist = torch.histc(pp, bins=255, min=0, max=255)
		nn_hist = torch.histc(nn, bins=255, min=0, max=255)
		pp_hist_flip = torch.flipud(pp_hist)
		nn_hist_flip = torch.flipud(nn_hist)
		pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
		nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

		precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
		recall = pp_hist_flip_cum / (gtNum + 1e-4)
		f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 1e-4)

		return dict(
			pre=torch.reshape(precision, (1, precision.shape[0])).cpu().data.numpy(),
			rec=torch.reshape(recall, (1, recall.shape[0])).cpu().data.numpy(),
			f1=torch.reshape(f1, (1, f1.shape[0])).cpu().data.numpy(),
		)
