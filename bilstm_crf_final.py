import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import os
import glob
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_classification_report
import logging
from tqdm import tqdm
import pickle
import random
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import re
from typing import List, Dict, Tuple
import fasttext
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_english_token(token: str) -> bool:
    """Heuristic to detect English tokens (Latin characters only)."""
    return bool(re.match(r'^[a-zA-Z0-9.,!?\'"()-]+$', token))

def load_pretrained_embeddings(khmer_path: str, english_path: str, word_to_idx: Dict[str, int], 
                              embedding_dim: int) -> np.ndarray:
    """Load pretrained Khmer and English FastText embeddings and create an embedding matrix."""
    logger.info(f"Loading Khmer embeddings from {khmer_path}")
    logger.info(f"Loading English embeddings from {english_path}")
    
    # Load FastText models directly
    try:
        khmer_model = fasttext.load_model(khmer_path)
        logger.info(f"Successfully loaded Khmer FastText model with {len(khmer_model.words)} words")
    except Exception as e:
        logger.error(f"Failed to load Khmer embeddings from {khmer_path}: {e}")
        raise
    
    try:
        english_model = fasttext.load_model(english_path)
        logger.info(f"Successfully loaded English FastText model with {len(english_model.words)} words")
    except Exception as e:
        logger.error(f"Failed to load English embeddings from {english_path}: {e}")
        raise
    
    # Verify embedding dimensions match
    khmer_dim = khmer_model.get_dimension()
    english_dim = english_model.get_dimension()
    
    if khmer_dim != embedding_dim:
        logger.warning(f"Khmer embedding dimension ({khmer_dim}) doesn't match expected ({embedding_dim})")
        embedding_dim = khmer_dim
    
    if english_dim != embedding_dim:
        logger.warning(f"English embedding dimension ({english_dim}) doesn't match expected ({embedding_dim})")
        embedding_dim = english_dim
    
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
    
    # Initialize <PAD> and <UNK> with special vectors
    embedding_matrix[word_to_idx['<PAD>']] = np.zeros(embedding_dim)
    embedding_matrix[word_to_idx['<UNK>']] = np.random.normal(0, 0.1, embedding_dim)
    
    # Map vocabulary to pretrained embeddings
    oov_count = 0
    english_token_count = 0
    khmer_token_count = 0
    
    for word, idx in word_to_idx.items():
        if word in ['<PAD>', '<UNK>']:
            continue
            
        try:
            if is_english_token(word):
                # For English tokens, try the English model first
                try:
                    embedding_matrix[idx] = english_model.get_word_vector(word)
                    english_token_count += 1
                except:
                    # If word not found, try lowercase
                    try:
                        embedding_matrix[idx] = english_model.get_word_vector(word.lower())
                        english_token_count += 1
                    except:
                        # If still not found, use random vector
                        embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
                        oov_count += 1
            else:
                # For Khmer tokens, use the Khmer model
                try:
                    embedding_matrix[idx] = khmer_model.get_word_vector(word)
                    khmer_token_count += 1
                except:
                    # If word not found, use random vector
                    embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
                    oov_count += 1
                    
        except Exception as e:
            logger.warning(f"Error processing word '{word}': {e}")
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
            oov_count += 1
    
    logger.info(f"Loaded pretrained embeddings. OOV words: {oov_count}/{len(word_to_idx)}")
    logger.info(f"English tokens loaded: {english_token_count}/{len(word_to_idx)}")
    logger.info(f"Khmer tokens loaded: {khmer_token_count}/{len(word_to_idx)}")
    
    return embedding_matrix

def build_vocab_and_tags(tokens: List[List[str]], tags: List[List[str]], min_freq: int = 2, 
                        khmer_path: str = None, english_path: str = None, 
                        embedding_dim: int = 300) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], np.ndarray]:
    """Build vocabulary and tag mappings, optionally with pretrained embeddings."""
    word_counter = Counter()
    tag_counter = Counter()
    
    for sentence, tag_list in zip(tokens, tags):
        word_counter.update(sentence)
        tag_counter.update(tag_list)
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.items():
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    tag_to_idx = {'<PAD>': 0}
    for tag in sorted(tag_counter.keys()):
        tag_to_idx[tag] = len(tag_to_idx)
    
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    # Load pretrained embeddings if provided
    embedding_matrix = None
    if khmer_path and english_path:
        embedding_matrix = load_pretrained_embeddings(khmer_path, english_path, word_to_idx, embedding_dim)
    
    logger.info(f"Vocabulary size: {len(word_to_idx)}")
    logger.info(f"Tag set size: {len(tag_to_idx)}")
    logger.info(f"Tags: {list(tag_to_idx.keys())}")
    
    return word_to_idx, tag_to_idx, idx_to_tag, embedding_matrix

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, dropout=0.5, 
                 tag_to_idx=None, pretrained_embeddings=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # Freeze embeddings; set to True for fine-tuning
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.tagset_size = tagset_size
        self.tag_to_idx = tag_to_idx

    def forward(self, sentences, mask):
        emissions = self._get_lstm_features(sentences, mask)
        return self._viterbi_decode(emissions, mask)

    def _get_lstm_features(self, sentences, mask):
        embeds = self.embedding(sentences)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True, total_length=embeds.size(1))
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, sentences, tags, mask):
        emissions = self._get_lstm_features(sentences, mask)
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        emissions = emissions.view(-1, self.tagset_size)
        tags = tags.view(-1)
        mask = mask.view(-1)
        weights = torch.ones(self.tagset_size, device=emissions.device)
        if self.tag_to_idx:
            for tag in self.tag_to_idx:
                if tag.startswith('I-'):
                    weights[self.tag_to_idx[tag]] = 5.0  # Higher weight for I- tags
        loss = nn.functional.cross_entropy(emissions[mask], tags[mask], weight=weights, reduction='none')
        return -loss

    def _viterbi_decode(self, emissions, mask):
        tag_seq = emissions.argmax(-1).cpu().numpy()
        return None, tag_seq
    
def load_json_files_robust(input_dir: str, verbose: bool = True) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """Load JSON files and extract tokens and BIO tags, adapted from finetune_xlmr.ipynb."""
    all_tokens = []
    all_tags = []
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not input_files:
        logger.error(f"No JSON files found in '{input_dir}'")
        return [], [], []
    
    logger.info(f"Processing {len(input_files)} files...")
    
    errors = []
    skipped_sentences = 0
    processed_sentences = 0
    
    for file_idx, input_file in enumerate(input_files):
        if verbose and file_idx < 5:
            logger.info(f"Processing: {os.path.basename(input_file)}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            
            processed_content = obj.get('processed_content', [])
            
            for sent_idx, sentence_data in enumerate(processed_content):
                try:
                    tokens = sentence_data.get('tokens', [])
                    bio_tags = sentence_data.get('bio_tags', [])
                    
                    if not tokens or not bio_tags:
                        skipped_sentences += 1
                        continue
                    
                    # Handle nested tag structure
                    flattened_tags = []
                    for tag in bio_tags:
                        if isinstance(tag, list):
                            flattened_tags.append(tag[0] if tag else "O")
                        else:
                            flattened_tags.append(tag if tag else "O")
                    
                    if len(tokens) != len(flattened_tags):
                        error_msg = f"Length mismatch in {os.path.basename(input_file)}, sentence {sent_idx}: {len(tokens)} tokens vs {len(flattened_tags)} tags"
                        errors.append(error_msg)
                        if verbose and len(errors) <= 3:
                            logger.warning(error_msg)
                        skipped_sentences += 1
                        continue
                    
                    # Validate tags
                    validated_tags = [tag if tag else "O" for tag in flattened_tags]
                    
                    all_tokens.append(tokens)
                    all_tags.append(validated_tags)
                    processed_sentences += 1
                    
                except Exception as e:
                    error_msg = f"Error processing sentence {sent_idx} in {os.path.basename(input_file)}: {e}"
                    errors.append(error_msg)
                    if verbose and len(errors) <= 3:
                        logger.error(error_msg)
                    skipped_sentences += 1
                    continue
        
        except Exception as e:
            error_msg = f"Error reading file {input_file}: {e}"
            errors.append(error_msg)
            if verbose:
                logger.error(error_msg)
            continue
    
    logger.info(f"\nData loading summary:")
    logger.info(f"  Processed sentences: {processed_sentences}")
    logger.info(f"  Skipped sentences: {skipped_sentences}")
    logger.info(f"  Total errors: {len(errors)}")
    
    if errors and verbose:
        logger.info(f"\nFirst few errors:")
        for error in errors[:5]:
            logger.info(f"    • {error}")
    
    return all_tokens, all_tags, input_files

class KhmerNERDataset(Dataset):
    """Dataset class for Khmer NER, adapted for BiLSTM-CRF."""
    def __init__(self, tokens: List[List[str]], tags: List[List[str]], 
                 word_to_idx: Dict[str, int], tag_to_idx: Dict[str, int], 
                 max_len: int = 128):
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_len = max_len
        self.data = self._load_and_process_data(tokens, tags)
    
    def _load_and_process_data(self, tokens: List[List[str]], tags: List[List[str]]) -> List[Tuple[List[int], List[int], List[bool]]]:
        data = []
        for sentence, tag_list in zip(tokens, tags):
            word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence]
            tag_indices = [self.tag_to_idx.get(tag, self.tag_to_idx['O']) for tag in tag_list]
            
            seq_len = min(len(word_indices), self.max_len)
            attention_mask = [True] * seq_len + [False] * (self.max_len - seq_len)
            
            if len(word_indices) < self.max_len:
                word_indices += [self.word_to_idx['<PAD>']] * (self.max_len - len(word_indices))
                tag_indices += [self.tag_to_idx['<PAD>']] * (self.max_len - len(tag_indices))
            else:
                word_indices = word_indices[:self.max_len]
                tag_indices = tag_indices[:self.max_len]
            
            data.append((word_indices, tag_indices, attention_mask))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words, tags, mask = self.data[idx]
        return (torch.tensor(words, dtype=torch.long),
                torch.tensor(tags, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool))

def split_data(all_tokens: List[List[str]], all_tags: List[List[str]]) -> Tuple:
    """Split data into train, dev, and test sets matching finetune_xlmr.ipynb."""
    # First split: 80% train, 20% temp
    train_tokens, temp_tokens, train_tags, temp_tags = train_test_split(
        all_tokens, all_tags, test_size=0.2, random_state=42, stratify=None
    )
    
    # Second split: 50% of temp to validation, 50% to test
    dev_tokens, test_tokens, dev_tags, test_tags = train_test_split(
        temp_tokens, temp_tags, test_size=0.5, random_state=42, stratify=None
    )
    
    logger.info(f"Data split - Train: {len(train_tokens)}, Dev: {len(dev_tokens)}, Test: {len(test_tokens)}")
    
    return train_tokens, dev_tokens, test_tokens, train_tags, dev_tags, test_tags

def prepare_data(input_dir: str, batch_size: int = 32, max_len: int = 128, min_freq: int = 2,
                 khmer_path: str = None, english_path: str = None, embedding_dim: int = 300) -> Tuple:
    all_tokens, all_tags, input_files = load_json_files_robust(input_dir)
    
    if not all_tokens:
        logger.error("No data loaded. Exiting.")
        return None, None, None, None, None, None, None
    
    train_tokens, dev_tokens, test_tokens, train_tags, dev_tags, test_tags = split_data(
        all_tokens, all_tags
    )
    
    # Add error handling for embedding loading
    try:
        word_to_idx, tag_to_idx, idx_to_tag, embedding_matrix = build_vocab_and_tags(
            all_tokens, all_tags, min_freq, khmer_path, english_path, embedding_dim
        )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        logger.info("Continuing without pretrained embeddings...")
        word_to_idx, tag_to_idx, idx_to_tag, embedding_matrix = build_vocab_and_tags(
            all_tokens, all_tags, min_freq, None, None, embedding_dim
        )
    
    train_dataset = KhmerNERDataset(train_tokens, train_tags, word_to_idx, tag_to_idx, max_len)
    dev_dataset = KhmerNERDataset(dev_tokens, dev_tags, word_to_idx, tag_to_idx, max_len)
    test_dataset = KhmerNERDataset(test_tokens, test_tags, word_to_idx, tag_to_idx, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, dev_loader, test_loader, word_to_idx, tag_to_idx, idx_to_tag, embedding_matrix

def train_model(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, 
                optimizer: optim.Optimizer, device: torch.device, epochs: int = 10,
                patience: int = 3, model_path: str = 'khmer_ner_best.pt') -> None:
    best_dev_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training')
        
        for words, tags, mask in train_pbar:
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            
            optimizer.zero_grad()
            loss = model.loss(words, tags, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for words, tags, mask in tqdm(dev_loader, desc='Validation'):
                words, tags, mask = words.to(device), tags.to(device), mask.to(device)
                dev_loss += model.loss(words, tags, mask).item()
        
        avg_dev_loss = dev_loss / len(dev_loader)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")
        
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_dev_loss
            }, model_path)
            patience_counter = 0
            logger.info(f"New best model saved with dev loss: {avg_dev_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break


logger = logging.getLogger(__name__)

def evaluate_model_manual(model: nn.Module, test_loader: DataLoader, idx_to_tag: Dict[int, str], 
                         device: torch.device, output_dir: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Evaluate the model, collecting predictions for token-level and entity-level metrics."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_true = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (words, tags, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            words, tags, mask = words.to(device), tags.to(device), mask.to(device)
            
            _, pred_tags = model(words, mask)
            
            for pred_seq, true_seq, mask_seq in zip(pred_tags, tags.cpu().numpy(), mask.cpu().numpy()):
                # Convert NumPy int64 to Python int and mask to Python bool
                pred_seq = [int(p) for p in pred_seq]
                true_seq = [int(t) for t in true_seq]
                mask_seq = [bool(m) for m in mask_seq]
                
                # Ensure pred_seq and true_seq are filtered using the same mask
                pred_filtered = [idx_to_tag.get(p, 'O') for p, m in zip(pred_seq, mask_seq) if m]
                true_filtered = [idx_to_tag.get(t, 'O') for t, m in zip(true_seq, mask_seq) if m]
                
                # Check for length mismatch
                if len(pred_filtered) != len(true_filtered):
                    logger.warning(f"Length mismatch in batch {batch_idx}: pred={len(pred_filtered)}, true={len(true_filtered)}")
                    continue
                
                all_preds.append(pred_filtered)
                all_true.append(true_filtered)
                all_predictions.append({
                    'true_tags': true_filtered,
                    'pred_tags': pred_filtered
                })
    
    logger.info(f"Evaluation completed on {len(all_predictions)} samples")
    
    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to {json_path}")
    
    conll_path = os.path.join(output_dir, "predictions.conll")
    with open(conll_path, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            for true_tag, pred_tag in zip(pred['true_tags'], pred['pred_tags']):
                f.write(f"{true_tag}\t{pred_tag}\n")
            f.write("\n")
    logger.info(f"CoNLL format saved to {conll_path}")
    
    return all_preds, all_true

def calculate_metrics(all_true_tags: List[List[str]], all_pred_tags: List[List[str]], output_dir: str) -> Dict:
    """Calculate token-level metrics, excluding 'O' tags, and overall accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for length mismatch at the sequence level
    if len(all_true_tags) != len(all_pred_tags):
        logger.error(f"Mismatch in number of sequences: true={len(all_true_tags)}, pred={len(all_pred_tags)}")
        return {'error': 'Mismatch in number of sequences'}
    
    # Flatten the lists of tags for token-level metric calculation
    flat_true_tags = []
    flat_pred_tags = []
    for true_seq, pred_seq in zip(all_true_tags, all_pred_tags):
        if len(true_seq) != len(pred_seq):
            logger.warning(f"Skipping sequence due to length mismatch: true={len(true_seq)}, pred={len(pred_seq)}")
            continue
        flat_true_tags.extend(true_seq)
        flat_pred_tags.extend(pred_seq)
    
    # Filter out 'O' and '<PAD>' tags for entity-focused token-level evaluation
    filtered_true_tags = []
    filtered_pred_tags = []
    for true_tag, pred_tag in zip(flat_true_tags, flat_pred_tags):
        if true_tag != 'O' and true_tag != '<PAD>':
            filtered_true_tags.append(true_tag)
            filtered_pred_tags.append(pred_tag)
    
    print("\n📊 Token-Level Evaluation Results:")
    print("=" * 60)
    print(f"Total tokens: {len(flat_true_tags)}")
    print(f"Entity tokens (excluding 'O' and '<PAD>'): {len(filtered_true_tags)}")
    print(f"Non-entity tokens ('O' or '<PAD>'): {len(flat_true_tags) - len(filtered_true_tags)}")
    print()
    
    metrics = {}
    if len(filtered_true_tags) > 0:
        # Get unique labels (excluding 'O' and '<PAD>')
        unique_labels = sorted(list(set(filtered_true_tags + filtered_pred_tags)))
        if 'O' in unique_labels:
            unique_labels.remove('O')
        if '<PAD>' in unique_labels:
            unique_labels.remove('<PAD>')
        
        # Calculate token-level metrics on entity tags only
        precision = precision_score(filtered_true_tags, filtered_pred_tags, 
                                  labels=unique_labels, average='weighted', zero_division=0)
        recall = recall_score(filtered_true_tags, filtered_pred_tags, 
                             labels=unique_labels, average='weighted', zero_division=0)
        f1 = f1_score(filtered_true_tags, filtered_pred_tags, 
                     labels=unique_labels, average='weighted', zero_division=0)
        
        print(f"Entity-Level Token Metrics (excluding 'O' tags):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()
        
        # Detailed classification report (entity tags only)
        print("Detailed Classification Report (Entity Tags Only):")
        report = classification_report(filtered_true_tags, filtered_pred_tags, 
                                     labels=unique_labels, zero_division=0)
        print(report)
        
        # Calculate overall token accuracy (including 'O' and '<PAD>' tags)
        overall_accuracy = sum(1 for t, p in zip(flat_true_tags, flat_pred_tags) if t == p) / len(flat_true_tags) if flat_true_tags else 0
        print(f"Overall Token-Level Accuracy (including 'O' tags): {overall_accuracy:.4f}")
        print()
        
        # Save token-level metrics, ensuring JSON-serializable types
        metrics = {
            'entity_precision': float(precision),
            'entity_recall': float(recall),
            'entity_f1_score': float(f1),
            'overall_accuracy': float(overall_accuracy),
            'total_tokens': int(len(flat_true_tags)),
            'entity_tokens': int(len(filtered_true_tags)),
            'non_entity_tokens': int(len(flat_true_tags) - len(filtered_true_tags)),
            'entity_classification_report': classification_report(
                filtered_true_tags, filtered_pred_tags, 
                labels=unique_labels, zero_division=0, output_dict=True
            )
        }
    else:
        print("❌ No entity tags found in the evaluation data!")
        metrics = {'error': 'No entity tags found'}
    
    # Save token-level metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Token-level metrics saved to {metrics_path}")
    
    return metrics

def calculate_entity_level_metrics(all_true_tags: List[List[str]], all_pred_tags: List[List[str]], output_dir: str) -> Dict:
    """Calculate entity-level precision, recall, and F1-score using seqeval library."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for length mismatch at the sequence level
    if len(all_true_tags) != len(all_pred_tags):
        logger.error(f"Mismatch in number of sequences: true={len(all_true_tags)}, pred={len(all_pred_tags)}")
        return {'error': 'Mismatch in number of sequences'}
    
    # Ensure no sequence-level length mismatches
    valid_true_tags = []
    valid_pred_tags = []
    for true_seq, pred_seq in zip(all_true_tags, all_pred_tags):
        if len(true_seq) != len(pred_seq):
            logger.warning(f"Skipping sequence due to length mismatch: true={len(true_seq)}, pred={len(pred_seq)}")
            continue
        valid_true_tags.append(true_seq)
        valid_pred_tags.append(pred_seq)
    
    if not valid_true_tags:
        logger.error("No valid sequences found after filtering mismatches")
        return {'error': 'No valid sequences found'}
    
    # Calculate entity-level metrics using seqeval
    precision = seqeval_precision(valid_true_tags, valid_pred_tags)
    recall = seqeval_recall(valid_true_tags, valid_pred_tags)
    f1 = seqeval_f1(valid_true_tags, valid_pred_tags)
    
    # Generate detailed classification report
    report = seqeval_classification_report(valid_true_tags, valid_pred_tags, output_dict=True, zero_division=0)
    
    print("\n📊 Entity-Level Evaluation Results (seqeval):")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Entity-Level Classification Report:")
    print(seqeval_classification_report(valid_true_tags, valid_pred_tags, zero_division=0))
    
    # Save metrics, ensuring JSON-serializable types
    metrics = {
        'entity_precision': float(precision),
        'entity_recall': float(recall),
        'entity_f1': float(f1),
        'classification_report': report
    }
    
    metrics_path = os.path.join(output_dir, "entity_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"💾 Entity metrics saved to {metrics_path}")
    
    return metrics

def save_vocabularies(word_to_idx: Dict[str, int], tag_to_idx: Dict[str, int], 
                     idx_to_tag: Dict[int, str], path: str = 'vocabularies.pkl') -> None:
    """Save vocabularies for later use."""
    with open(path, 'wb') as f:
        pickle.dump({
            'word_to_idx': word_to_idx,
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag
        }, f)
    logger.info(f"Vocabularies saved to {path}")

def main():
    """Main execution function."""
    # Configuration
    config = {
        'data_folder': "/home/guest/Public/KHEED/KHEED_Data_Collection/Final/bio_tagged",
        'output_folder': "/home/guest/Public/KHEED/KHEED_Data_Collection/Evaluation/bilstm_crf_ner_model",
        'khmer_embeddings': "/home/guest/Public/KHEED/KHEED_Data_Collection/cc.km.300.bin",  # Binary format
        'english_embeddings': "/home/guest/Public/KHEED/KHEED_Data_Collection/cc.en.300.bin",  # Binary format
        'train_ratio': 0.8,
        'dev_ratio': 0.1,
        'test_ratio': 0.1,
        'random_seed': 42,
        'batch_size': 32,
        'max_len': 128,
        'min_freq': 2,
        'embedding_dim': 300,  # Match fastText embedding dimension
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 100,
        'patience': 3,
        'model_path': 'khmer_ner_best.pt',
        'vocab_path': 'vocabularies.pkl'
    }

    # Add file validation
    if not os.path.exists(config['khmer_embeddings']):
        logger.error(f"Khmer embeddings file not found: {config['khmer_embeddings']}")
        return
    
    if not os.path.exists(config['english_embeddings']):
        logger.error(f"English embeddings file not found: {config['english_embeddings']}")
        return
    
    # Check if files are actually binary
    try:
        with open(config['khmer_embeddings'], 'rb') as f:
            header = f.read(4)
            if header[:2] != b'\x1f\x8b':  # Check if it's gzipped
                logger.info("Khmer embeddings appear to be in binary format")
    except Exception as e:
        logger.warning(f"Could not verify Khmer embeddings format: {e}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Loading and preparing data...")
    train_loader, dev_loader, test_loader, word_to_idx, tag_to_idx, idx_to_tag, embedding_matrix = prepare_data(
        config['data_folder'], config['batch_size'], config['max_len'], config['min_freq'],
        khmer_path=config['khmer_embeddings'], english_path=config['english_embeddings'], 
        embedding_dim=config['embedding_dim']
    )
    
    if train_loader is None:
        logger.error("Failed to prepare data. Exiting.")
        return
    
    # Save vocabularies
    save_vocabularies(word_to_idx, tag_to_idx, idx_to_tag, config['vocab_path'])
    
    # Initialize model
    model = BiLSTM_CRF(
        vocab_size=len(word_to_idx),
        tagset_size=len(tag_to_idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        tag_to_idx=tag_to_idx,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Train model
    logger.info("Starting model training...")
    train_model(model, train_loader, dev_loader, optimizer, device, 
                config['epochs'], config['patience'], config['model_path'])
    
    # Evaluate model
    logger.info("Starting model evaluation...")
    all_preds, all_true = evaluate_model_manual(model, test_loader, idx_to_tag, device, config['output_folder'])
    
    # Calculate token-level metrics
    logger.info("Calculating token-level metrics...")
    token_metrics = calculate_metrics(all_true, all_preds, config['output_folder'])
    
    # Calculate entity-level metrics
    logger.info("Calculating entity-level metrics...")
    entity_metrics = calculate_entity_level_metrics(all_true, all_preds, config['output_folder'])
    
    # Save final metrics
    final_metrics = {
        'token_level_metrics': token_metrics,
        'entity_level_metrics': entity_metrics
    }
    final_metrics_path = os.path.join(config['output_folder'], "final_metrics.json")
    with open(final_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Final metrics saved to {final_metrics_path}")
    
if __name__ == "__main__":
    main()