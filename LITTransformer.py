import logging
import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from lion_pytorch import Lion
from functools import partial

# Local Imports
from dataset import BilingualDataset
from train import get_model, get_or_build_tokenizer, greedy_decode

logger = logging.getLogger("Transformer")
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(filename="prediction.log")
file_handler.setLevel(level=logging.INFO)
logger.addHandler(file_handler)

class LITTransformer(pl.LightningModule):
    """
    PyTorch Lightning Code for Transformer
    """

    def __init__(self, config: dict):
        """
        Constructor
        :param config: Dictionary with training parameter configuration
        """
        # Initialize lightning module
        super().__init__()

        # Prepare Data and Model
        self.config = config
        self.prepare_data()
        self.tokenizer_src = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = get_or_build_tokenizer(self.config, self.ds_raw, self.config['lang_tgt'])
        self.model = get_model(self.config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size())

        # Initialize variables
        self.ds_raw = None
        self.train_ds = None
        self.val_ds = None

        self.loss_fn = None
        self.train_loss = []

        self.source_texts = []
        self.expected = []
        self.predicted = []

    def forward(self, x):
        """
        Forward pass for the model training
        :param x: Inputs
        """
        return self.model(x)

    # ##################################################################################################
    # ############################## Training Configuration Related Hooks ##############################
    # ##################################################################################################
    def configure_optimizers(self):
        """
        Method to configure the optimizer
        """
        optimizer = Lion(self.parameters(), lr=1e-4/10, weight_decay=1e-2)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """
        Configure gradients resetting technique for optimizer
        :param epoch: Number of epoch
        :param batch_idx: Batch Number
        :param optimizer: Optimizer Used
        """
        # Set gradients to None instead of zero, helps to improve memory usage
        optimizer.zero_grad(set_to_none=True)

    def get_loss_fn(self):
        """
        Method to return loss function after the parameters are initialized
        """
        if not self.loss_fn:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(self.device)
        return self.loss_fn

    # #################################################################################################
    # ################################## Training Loop Related Hooks ##################################
    # #################################################################################################
    def training_step(self, train_batch, batch_index):
        """
        Method called on training dataset to train the model
        :param train_batch: Batch containing images and labels
        :param batch_index: Index of the batch
        """
        # Input and mask for the encoder and decoder
        encoder_input = train_batch['encoder_input'].to(self.device)  # (b, seq_len)
        decoder_input = train_batch['decoder_input'].to(self.device)  # (B, seq_len)
        encoder_mask = train_batch['encoder_mask'].to(self.device)    # (B, 1, 1, seq_len)
        decoder_mask = train_batch['decoder_mask'].to(self.device)    # (B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask)                                # (B, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (
        proj_output = self.model.project(decoder_output)                                               # (B, seq_len, vocab_size)

        # Compare the output with the label
        label = train_batch['label'].to(self.device)  # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss_fn = self.get_loss_fn()
        loss = loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))

        # Log the data to visualize it using Tensorboard
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.train_loss.append(loss)
        return loss

    def on_train_epoch_end(self):
        """
        Method called after every training epoch
        """
        self.log('loss', torch.stack(self.train_loss).mean(), on_epoch=True, logger=True)
        print(f"Loss Mean - {torch.stack(self.train_loss).mean()}")
        self.train_loss.clear()
        garbage_collection_cuda()

    def evaluate(self, batch, stage=None):
        """
        Common logic for validation and test
        :param batch: Input data batch
        :param stage: test/val
        """
        encoder_input = batch['encoder_input'].to(self.device)  # (b, seq_len)
        encoder_mask = batch['encoder_mask'].to(self.device)    # (b, 1, 1, seq_len)

        model_out = greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.config['seq_len'], self.device)

        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

        logger.info(f"{f'SOURCE: ': >12}{source_text}")
        logger.info(f"{f'TARGET: ': >12}{target_text}")
        logger.info(f"{f'PREDICTED: ': >12}{model_out_text}")
        logger.info("-" * 20)
        logger.info("  ")

    def validation_step(self, batch, batch_idx):
        """
        Method called on validation dataset to check if the model is learning
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        self.evaluate(batch, "val")

    def on_validation_epoch_end(self):
        """
        Method to be called at the end of the validation epoch
        """
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(self.predicted, self.expected)
        self.log('validation cer', cer, prog_bar=True, on_epoch=True, logger=True)

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(self.predicted, self.expected)
        self.log('validation wer', wer, prog_bar=True, on_epoch=True, logger=True)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(self.predicted, self.expected)
        self.log('validation BLEU', bleu, prog_bar=True, on_epoch=True, logger=True)

        # Clear the data
        self.source_texts = []
        self.expected = []
        self.predicted = []
        garbage_collection_cuda()

    def test_step(self, batch, batch_idx):
        """
        Method called on test dataset to check model performance on unseen data
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        return self.evaluate(batch, "test")

    # ##############################################################################################
    # ##################################### Data Related Hooks #####################################
    # ##############################################################################################

    def prepare_data(self):
        """
        Method to download the dataset
        """
        # It only has the train split, so we divide it ourselves
        self.ds_raw = load_dataset('opus_books', f"{self.config['lang_src']}-{self.config['lang_tgt']}", split='train')

    def setup(self, stage=None):
        """
        Method to create Split the dataset into train, test and val
        """
        # Get tokenizers
        tokenizer_src = self.tokenizer_src
        tokenizer_tgt = self.tokenizer_tgt
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")],dtype=torch.int64)
        # Keep 90% for training, 10% for validation
        train_ds_size = int(0.9 * len(self.ds_raw))
        val_ds_size = len(self.ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(self.ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                         self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len'])
        self.val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                                       self.config['lang_src'], self.config['lang_tgt'], self.config['seq_len'])

        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        for item in self.ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

    def train_dataloader(self):
        """
        Method to return the DataLoader for Training set
        """
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, collate_fn=partial(self.collate_batch,self.pad_token))

    def val_dataloader(self):
        """
        Method to return the DataLoader for the Validation set
        """
        return DataLoader(self.val_ds, batch_size=1, shuffle=True, collate_fn=partial(self.collate_batch,self.pad_token))
    
    def collate_batch(self, pad_token, batch):
        """
        Method to preprocess each batch of the subset
        """
        def pad_sequence(seq, max_len):
            return torch.cat([
                seq,
                torch.tensor([pad_token] * (max_len - seq.size(0)), dtype=torch.int64)
            ], dim=0)
        def causal_mask(size):
            mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
            return mask == 0
    
        encoder_lengths = [item['encoder_input'].size(0) for item in batch]
        decoder_lengths = [item['decoder_input'].size(0) for item in batch]
        max_seq_len = max(encoder_lengths + decoder_lengths)

        for item in batch:
            item['encoder_input'] = pad_sequence(item['encoder_input'], max_seq_len)
            item['decoder_input'] = pad_sequence(item['decoder_input'], max_seq_len)
            item['label'] = pad_sequence(item['label'], max_seq_len)

        encoder_inputs = torch.stack([item['encoder_input'] for item in batch])
        decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])

        encoder_masks = torch.stack([
            (item['encoder_input'] != pad_token).unsqueeze(0).unsqueeze(1).int()
            for item in batch
        ])

        decoder_masks = torch.stack([
            (item['decoder_input'] != pad_token).int() & causal_mask(item['decoder_input'].size(-1))
            for item in batch
        ])

        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]

        return {
            'encoder_input': encoder_inputs,
            'decoder_input': decoder_inputs,
            'label': labels,
            'encoder_mask': encoder_masks,
            'decoder_mask': decoder_masks,
            'src_text': src_texts,
            'tgt_text': tgt_texts
        }
