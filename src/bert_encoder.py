# src/bert_encoder.py

from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class BertEncoder:
    """
    한국어 BERT 기반 문서 임베딩 생성기.

    기본값은 klue/bert-base 이지만,
    main에서 인자로 다른 모델 이름을 넘겨도 됨.
    """

    def __init__(
        self,
        model_name: str = "klue/bert-base",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        입력 텍스트 리스트를 BERT [CLS] 임베딩으로 변환.

        반환:
          - shape: (len(texts), hidden_size) 의 numpy 배열
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            # 빈 문자열 방어
            batch_texts = [t if t.strip() else "[PAD]" for t in batch_texts]

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc)
            # last_hidden_state: (batch, seq_len, hidden)
            # [CLS] 토큰은 보통 index 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
            cls_embeddings = cls_embeddings.cpu().numpy()

            all_embeddings.append(cls_embeddings)

        if not all_embeddings:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings
