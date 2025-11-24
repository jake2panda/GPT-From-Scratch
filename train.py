
def token_ids_to_text(idxs, encoding):
	decoded_text = encoding.decode(idxs.squeeze(0).tolist())
	return decoded_text


def text_to_token_ids(text, encoding):
	encoded_text = encoding.encode(text)
	return torch.tensor(encoded_text).unsqueeze(0)


def calc_loss_batch(input_batch, target_batch, model):
	logits = model(input_batch)
	loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())

	return loss


def calc_loss_loader(data_loader, model, num_batches=None, device=None):
	total_loss = 0.
	if len(data_loader) == 0:
		return float("nan")
	elif num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))

	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			input_batch = input_batch.to(device)
			target_batch = target_batch.to(device)
			loss = calc_loss_batch(input_batch, target_batch, model)
			total_loss += loss.item()
		else:
			break


	return total_loss / num_batches




def evaluate_model(model, train_loader, val_loader, eval_iter, device):
	model.eval()
	with torch.no_grad():
		train_loss = calc_loss_loader(
			train_loader, model, num_batches=eval_iter,device=device
		)
		val_loss = calc_loss_loader(
			val_loader, model, num_batches=eval_iter, device=device
		)
	model.train()

	return train_loss, val_loss


def generate_text_simple(model, prompt_token, max_new_tokens, context_size, temperature=0.7):
	current_tokens = prompt_token.clone()
	for _ in range(max_new_tokens):
		context_token_idx = current_tokens[:, -context_size:]
		with torch.no_grad():
			logits = model(context_token_idx)

		next_token_logits = logits[:, -1, :]
		next_token_logits = next_token_logits / temperature

		probs = torch.softmax(next_token_logits, dim=-1)
		token_idx = torch.multinomial(probs, num_samples=1)
		current_tokens = torch.cat([current_tokens, token_idx], dim=1)

	return current_tokens


def generate_and_print_sample(model, tokenizer, start_context, device):
	model.eval()
	context_size = model.pos_emb.weight.shape[0]
	encoded = text_to_token_ids(start_context,tokenizer).to(device)
	with torch.no_grad():
		token_ids = generate_text_simple(model=model,prompt_token=encoded, max_new_tokens=50,context_size=context_size)

	decoded_text = token_ids_to_text(token_ids.cpu(), tokenizer)
	print(decoded_text.replace("\n", " "))
	#print(decoded_text)
	model.train()




def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
	train_losses, val_losses, track_tokens_seen = [],[],[]
	token_seen, global_step = 0, -1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	for epoch in range(num_epochs):
		model.train()
		for input_batch,target_batch in train_loader:
			optimizer.zero_grad()
			loss = calc_loss_batch(input_batch.to(device), target_batch.to(device), model)
			loss.backward()
			optimizer.step()

			token_seen += input_batch.numel()
			global_step += 1

			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(model, train_loader, val_loader, eval_iter, device)
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				track_tokens_seen.append(token_seen)
				print(f"Ep {epoch + 1} (step {global_step:06d}) : train_loss {train_loss:.4f} val_loss : {val_loss:.4f}")

		generate_and_print_sample(model, tokenizer, start_context, device)

	return train_losses, val_losses, track_tokens_seen


# def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs=10, 
#                       eval_freq=5, eval_iter=5, start_context="", tokenizer=None):
    
#     # Move model to GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model = model.to(device)
    
#     train_losses = []
#     val_losses = []
#     token_sees = []
    
#     criterion = nn.CrossEntropyLoss()
    
#     for epoch in range(num_epochs):
#         # Training
#         model.train()
#         total_train_loss = 0
        
#         for batch_idx, batch in enumerate(train_loader):
#             # Move batch to GPU
#             input_ids = batch['input_ids'].to(device)
#             labels = batch['labels'].to(device)
            
#             optimizer.zero_grad()
#             outputs = model(input_ids)
#             loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
#             loss.backward()
#             optimizer.step()
            
#             total_train_loss += loss.item()
        
#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation
#         if epoch % eval_freq == 0:
#             model.eval()
#             total_val_loss = 0
            
#             with torch.no_grad():
#                 for batch_idx, batch in enumerate(val_loader):
#                     if batch_idx >= eval_iter:  # Limit validation batches
#                         break
                    
#                     # Move batch to GPU
#                     input_ids = batch['input_ids'].to(device)
#                     labels = batch['labels'].to(device)
                    
#                     outputs = model(input_ids)
#                     loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
#                     total_val_loss += loss.item()
            
#             avg_val_loss = total_val_loss / min(eval_iter, len(val_loader))
#             val_losses.append(avg_val_loss)
            
#             # Generate text
#             generated_text = generate_token(model, start_context, max_len=20, 
#                                           context_len=CONFIG["context_length"], 
#                                           tokenizer=tokenizer, device=device)
#             token_sees.append(generated_text)
            
#             print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
#             print(f"Generated: {generated_text}")
    
#     return train_losses, val_losses, token_sees



import jakedataload
import torch
import tiktoken
from config import GPT_CONFIG
import gpt_from_scratch
import torch.nn as nn


tokenizer = tiktoken.encoding_for_model("gpt2")


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	text_data = f.read()


train_ratio = 0.8

split_index = int(train_ratio * len(text_data))

train_data = text_data[:split_index]
val_data = text_data[split_index:]


train_loader = jakedataload.create_dataloader(
	train_data,
	batch_size=2,
	max_len=GPT_CONFIG["context_length"],
	stride=GPT_CONFIG["context_length"],
	drop_last=False,
	shuffle=True,
	num_workers=0
	)

val_loader = jakedataload.create_dataloader(
	val_data,
	batch_size=2,
	max_len=GPT_CONFIG["context_length"],
	stride=GPT_CONFIG["context_length"],
	drop_last=False,
	shuffle=True,
	num_workers=0
	)


torch.manual_seed(123)

model = gpt_from_scratch.GPT_Model(GPT_CONFIG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)

model = model.to(device)

optimizer = torch.optim.AdamW(
     model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 1



train_losses, val_losses, token_sees = train_model_simple(
	model, train_loader, val_loader, optimizer,
	num_epochs=num_epochs, eval_freq=5, eval_iter=5,
	start_context="Hello, I am here to ", tokenizer=tokenizer
)








