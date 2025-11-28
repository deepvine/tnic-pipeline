# src/bert_encoder.py

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class BertEncoder:
    model_name: str = "skt/kobert-base-v1"
    device: Optional[str] = None
    batch_size: int = 16
    max_length: int = 256
    pooler: str = "cls"  # "cls" 또는 "mean"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        입력: 텍스트 리스트
        출력: numpy 배열 (num_texts, hidden_size)
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)  # last_hidden_state: (B, L, H)

            if self.pooler == "cls":
                # [CLS] 토큰 벡터 사용
                emb = outputs.last_hidden_state[:, 0, :]  # (B, H)
            elif self.pooler == "mean":
                # attention_mask 기반 mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                masked_hidden = outputs.last_hidden_state * mask
                sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
                lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                emb = sum_hidden / lengths
            else:
                raise ValueError(f"Unknown pooler: {self.pooler}")

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)
