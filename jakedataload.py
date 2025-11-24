import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
	def __init__(self, txt, tokenizer, max_len, stride):
		self.input_ids = []
		self.target_ids = []

		token_ids = tokenizer.encode(txt)

		for i in range(0, len(token_ids) - max_len, stride):
			input_chunk = token_ids[i:i + max_len]
			target_chunk = token_ids[i + 1: i + 1 + max_len]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_len=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
	tokenizer = tiktoken.encoding_for_model("gpt2")
	dataset = GPTDataset(txt, tokenizer, max_len, stride)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
		num_workers=num_workers
	)
	return dataloader



# with open("the-verdict.txt", "r", encoding="utf-8") as f:
# 	raw_text = f.read()


# dataloader = create_dataloader(raw_text, batch_size=2, max_len=6, stride=1, shuffle=False)

# da_iter = iter(dataloader)

# inputs, target = next(da_iter)

# print(next(da_iter))

# print(inputs.shape)
# print(target.shape)























