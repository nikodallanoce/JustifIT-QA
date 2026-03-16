import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen3Reranker:
    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-Reranker-4B",
            instruction: str = "Given an Italian legal question, retrieve relevant statutory texts that answer the query.",
            max_length: int = 4096,
            batch_size: int = 8,
            cache_dir: str = "/disk1/n.dallanoce/models",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device  # "cuda" if torch.cuda.is_available() else "cpu"
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True, "dtype": "auto"}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        ).to(self.device).eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        if self.token_true_id is None or self.token_false_id is None:
            raise ValueError("Impossibile risolvere i token ids di 'yes' e 'no'.")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _tokenize_pairs(self, formatted_pairs: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            formatted_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        for i, ids in enumerate(enc["input_ids"]):
            enc["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        batch = self.tokenizer.pad(
            enc,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        all_scores = []

        for start in range(0, len(documents), self.batch_size):
            docs_batch = documents[start: start + self.batch_size]
            formatted = [self._format_pair(query, doc) for doc in docs_batch]
            inputs = self._tokenize_pairs(formatted)

            logits = self.model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]

            yes_no_logits = torch.stack([false_logits, true_logits], dim=1)
            probs = torch.nn.functional.softmax(yes_no_logits, dim=1)[:, 1]
            all_scores.extend(probs.detach().float().cpu().tolist())

        return all_scores

    def rerank(self, query: str, documents: list[str], return_scores: bool = False):
        scores = np.asarray(self.score(query, documents), dtype=float)
        order = np.argsort(-scores)

        if return_scores:
            return [(documents[i], float(scores[i])) for i in order]
        return [documents[i] for i in order]
