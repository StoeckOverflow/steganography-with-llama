from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

def get_bert_next_token_probability_and_logits(model, tokenizer, text, initial_context_length=3, batch_size=32):
    token_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)
    max_length = len(token_ids[0])  # Maximum length for padding
    true_token_probabilities = []
    softmax_logits = []

    for start_index in range(initial_context_length, len(token_ids[0]) - 1, batch_size):
        end_index = min(start_index + batch_size, len(token_ids[0]) - 1)

        batch_token_ids = []
        for masked_index in range(start_index, end_index):
            truncated_token_ids = token_ids.clone()
            truncated_token_ids[0][masked_index] = tokenizer.mask_token_id
            truncated_token_ids = truncated_token_ids[:, :masked_index+1]

            # Pad the sequence to the maximum length
            padding_length = max_length - truncated_token_ids.shape[1]
            if padding_length > 0:
                padding_ids = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long)
                truncated_token_ids = torch.cat([truncated_token_ids, padding_ids], dim=1)

            batch_token_ids.append(truncated_token_ids)

        # Convert list of tensors to a single tensor
        batch_token_ids = torch.cat(batch_token_ids, dim=0)

        with torch.no_grad():
            outputs = model(batch_token_ids)
            predictions = outputs[0]

            for i in range(len(batch_token_ids)):
                logits = predictions[i, -1].cpu().numpy()  # Last token in the sequence
                softmax_logits.append(logits)
                predicted_prob = torch.nn.functional.softmax(predictions[i, -1], dim=-1)
                true_token_prob = predicted_prob[token_ids[0][start_index + i]].item()
                true_token_probabilities.append(true_token_prob)

    return true_token_probabilities, softmax_logits