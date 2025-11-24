import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from config import GPT_CONFIG


class MultiheadAttention(nn.Module):
	def __init__(self,CONFIG):
		super().__init__()
		assert(CONFIG["tok_emb"] % CONFIG["num_heads"] == 0)
		self.head_dim = CONFIG["tok_emb"] // CONFIG["num_heads"]
		self.tok_emb = CONFIG["tok_emb"]
		self.num_heads = CONFIG["num_heads"]
		self.w_query = nn.Linear(self.tok_emb, self.tok_emb, bias=CONFIG["ik_bias"])
		self.w_key = nn.Linear(self.tok_emb, self.tok_emb, bias=CONFIG["ik_bias"])
		self.w_value = nn.Linear(self.tok_emb, self.tok_emb, bias=CONFIG["ik_bias"])
		self.dropout = nn.Dropout(CONFIG["dropout"])
		self.out_logit = nn.Linear(self.tok_emb, self.tok_emb)
		self.register_buffer(
			"jmask",
			torch.triu(torch.ones(CONFIG["context_length"], CONFIG["context_length"]), diagonal=1)
		)

	def forward(self, x):
		b, seq_len, emb_dim = x.shape
		queries = self.w_query(x)
		keys = self.w_key(x)
		values = self.w_value(x)

		queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
		keys = keys.view(b, seq_len, self.num_heads, self.head_dim)
		values = values.view(b, seq_len, self.num_heads, self.head_dim)

		queries = queries.transpose(1,2)
		keys = keys.transpose(1,2)
		values = values.transpose(1,2)

		att_score = queries @ keys.transpose(2,3)
		masked = att_score.masked_fill(self.jmask.bool()[:seq_len, :seq_len], -torch.inf)
		att_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
		att_dropout = self.dropout(att_weights)
		context_vec = (att_dropout @ values).transpose(1,2)
		context_vec = context_vec.contiguous().view(b, seq_len, self.tok_emb)
		logit = self.out_logit(context_vec)
		return logit


class FeedForwardLayer(nn.Module):
	def __init__(self, CONFIG):
		super().__init__()
		self.layer1 = nn.Linear(CONFIG["tok_emb"], 4 * CONFIG["tok_emb"])
		self.layer2 = nn.Linear(4 * CONFIG["tok_emb"], CONFIG["tok_emb"])
		self.ffd = nn.Sequential(
			self.layer1,
			nn.GELU(),
			nn.Dropout(CONFIG["dropout"]),
			self.layer2
		)

	def forward(self, x):
		ffl_out = self.ffd(x)
		return ffl_out


class LayerNormalization(nn.Module):
	def __init__(self, CONFIG):
		super().__init__()
		self.layer_norm = nn.LayerNorm(
			normalized_shape=CONFIG["tok_emb"],
			elementwise_affine=True, # this enable scale and shift factor
			eps=1e-5
		)

	def forward(self, x):
		ln_out = self.layer_norm(x)
		return ln_out


class TransformerBlock(nn.Module):
	def __init__(self, CONFIG):
		super().__init__()
		self.mhat = MultiheadAttention(CONFIG)
		self.ln = LayerNormalization(CONFIG)
		self.ffl_layer = FeedForwardLayer(CONFIG)
		self.dropout = nn.Dropout(CONFIG["dropout"])

	def forward(self, x):

		shortcut = x
		x = self.ln(x)
		x = self.mhat(x)
		x = self.dropout(x)
		x = x + shortcut

		shortcut = x

		x = self.ln(x)
		x = self.ffl_layer(x)
		x = self.dropout(x)
		x = x + shortcut

		return x




class GPT_Model(nn.Module):
	def __init__(self, CONFIG):
		super().__init__()
		self.tok_emb = nn.Embedding(CONFIG["vocab_size"], CONFIG["tok_emb"])
		self.pos_emb = nn.Embedding(CONFIG["context_length"], CONFIG["tok_emb"])
		self.dropout = nn.Dropout(CONFIG["dropout"])
		self.transformerblock = nn.Sequential(
			*[TransformerBlock(CONFIG) for _ in range(CONFIG["n_layer"])]
		)
		self.ln = LayerNormalization(CONFIG)
		self.Linear_out = nn.Linear(CONFIG["tok_emb"], CONFIG["vocab_size"])


	def forward(self, x):
		b, num_tokens = x.shape
		tok_emb = self.tok_emb(x)
		pos_emb = self.pos_emb(torch.arange(num_tokens, device=x.device))
		x = tok_emb + pos_emb
		x = self.dropout(x)

		x = self.transformerblock(x)
		x = self.ln(x)
		x = self.Linear_out(x)

		return x

	# def generate_token(self, model, prompt_token, max_len, context_len):
	# 	for _ in range(max_len):
	# 		context_token_idx = prompt_token[:, -context_len:]
	# 		with torch.no_grad():
	# 			logits = model(context_token_idx)
	# 		logits = logits[:, -1, :]
	# 		prob = torch.softmax(logits, dim=-1)
	# 		predicted_token = torch.argmax(prob, dim=-1, keepdim=True)
	# 		prompt_token = torch.cat((prompt_token, predicted_token), dim=-1)

	# 	return prompt_token

	def generate_token(self,model, prompt_token, max_len, context_len, temperature=0.7):
		current_tokens = prompt_token.clone()
		for _ in range(max_len):
			context_token_idx = current_tokens[:, -context_len:]
			with torch.no_grad():
				logits = model(context_token_idx)

			next_token_logits = logits[:, -1, :]
			next_token_logits = next_token_logits / temperature

			probs = torch.softmax(next_token_logits, dim=-1)
			token_idx = torch.multinomial(probs, num_samples=1)
			current_tokens = torch.cat([current_tokens, token_idx], dim=1)


		return current_tokens




# import tiktoken
# tokenizer = tiktoken.encoding_for_model("gpt2")

# txt1 = tokenizer.encode("this is jake hammer")
# txt2 = tokenizer.encode("final text later is")
# txt3 = tokenizer.encode("day is off now")
# txt4 = tokenizer.encode("good code always win everytime")


# tensors = [torch.tensor(txt1), torch.tensor(txt2), torch.tensor(txt3), torch.tensor(txt4)]
# batch = rnn_utils.pad_sequence(tensors, batch_first=True, padding_value=0)


# model = GPT_Model(GPT_CONFIG)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model =model.to(device)

# torch.manual_seed(12)

# ask_question = "cogent way to call "

# prompt = tokenizer.encode(ask_question)

# p = torch.tensor(tokenizer.encode(ask_question)).unsqueeze(0)

# max_len = 10
# context_len = 1024



# model.eval()

# idx = model.generate_token(model, p, max_len, context_len)


# print(tokenizer.decode(idx.squeeze(0).tolist()))

