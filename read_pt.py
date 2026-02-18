import argparse
import torch
import collections


def summarize_tensor(tensor, max_elems=10):
	try:
		numel = tensor.numel()
	except Exception:
		print('      <non-tensor value>', type(tensor))
		return
	print(f'      shape={tuple(tensor.shape)}, dtype={tensor.dtype}, numel={numel}')
	if numel <= max_elems:
		print(f'      values={tensor.detach().cpu().numpy()}')
	else:
		vals = tensor.detach().cpu().float()
		print(f'      mean={float(vals.mean()):.6g}, std={float(vals.std()):.6g}')


def inspect_checkpoint(path):
	chk = torch.load(path, map_location='cpu')

	# If a raw model object was saved, try to handle it
	if not isinstance(chk, collections.abc.Mapping):
		print('Loaded object is not a dict. Type:', type(chk))
		if hasattr(chk, 'state_dict'):
			print('Object has state_dict(); summarizing parameters:')
			state = chk.state_dict()
			for n, v in state.items():
				print('  param:', n)
				summarize_tensor(v)
		return

	print('Checkpoint keys:', list(chk.keys()))

	# Common keys that may hold metrics/params
	for key, val in chk.items():
		kl = key.lower()
		print(f'Key: "{key}"  type: {type(val)}')

		if kl in ('state_dict', 'model_state', 'model_state_dict') or isinstance(val, dict) and all(hasattr(x, 'shape') or torch.is_tensor(x) for x in val.values()):
			print('  Interpreting as parameter state dict:')
			for name, tensor in list(val.items())[:200]:
				print('  -', name)
				summarize_tensor(tensor)

		elif 'metric' in kl or 'history' in kl or isinstance(val, (list, tuple)) and all(isinstance(x, (int, float, dict)) for x in val):
			print('  Interpreting as metrics/history:')
			print('  ', val)

		elif isinstance(val, dict):
			# print small dicts (like params)
			if len(val) <= 50:
				print('  dict contents:')
				for subk, subv in val.items():
					print('   -', subk, type(subv))
					if torch.is_tensor(subv):
						summarize_tensor(subv)
					else:
						print('      ', subv)
			else:
				print('  (large dict; skipping detailed print)')

		elif torch.is_tensor(val):
			print('  Tensor value:')
			summarize_tensor(val)

		else:
			print('  Value repr:', repr(val)[:400])


def main():
	parser = argparse.ArgumentParser(description='Inspect a PyTorch .pt checkpoint and print metrics/parameters')
	parser.add_argument('path', nargs='?', default='best_model.pt', help='Path to .pt file (default: best_model.pt)')
	args = parser.parse_args()

	inspect_checkpoint(args.path)


if __name__ == '__main__':
	main()
