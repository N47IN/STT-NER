import json
from typing import List, Dict, Any
from torch.utils.data import Dataset


class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, label_list: List[str], max_length: int = 256, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.is_train = is_train

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])

                char_tags = ["O"] * len(text)
                for e in entities:
                    s, e_idx, lab = e["start"], e["end"], e["label"]
                    if s < 0 or e_idx > len(text) or s >= e_idx:
                        continue
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e_idx):
                        char_tags[i] = f"I-{lab}"

                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )
                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                bio_tags = []
                for (start, end) in offsets:
                    if start == end:
                        # Special tokens ([CLS], [SEP], [PAD])
                        bio_tags.append("O")
                    else:
                        # Check all characters in the token span for entity labels
                        # Priority: B- tags first, then I- tags, then O
                        token_labels = []
                        for char_idx in range(start, min(end, len(char_tags))):
                            if char_idx < len(char_tags):
                                token_labels.append(char_tags[char_idx])
                        
                        # Find the most appropriate label
                        # Priority: B- > I- > O (prefer entity labels)
                        b_labels = [l for l in token_labels if l.startswith("B-")]
                        i_labels = [l for l in token_labels if l.startswith("I-")]
                        
                        if b_labels:
                            # Use the first B- label
                            bio_tags.append(b_labels[0])
                        elif i_labels:
                            # Use the first I- label
                            bio_tags.append(i_labels[0])
                        else:
                            # All are "O"
                            bio_tags.append("O")

                if len(bio_tags) != len(input_ids):
                    bio_tags = ["O"] * len(input_ids)

                label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

                self.items.append(
                    {
                        "id": obj["id"],
                        "text": text,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": label_ids,
                        "offset_mapping": offsets,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    def pad(seq, pad_value, max_len):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id, max_len) for ids in input_ids_list]
    attention_mask = [pad(am, 0, max_len) for am in attention_list]
    labels = [pad(lab, label_pad_id, max_len) for lab in labels_list]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
    return out
