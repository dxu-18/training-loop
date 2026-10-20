"""torch-lite torch.nn.utils.rnn: pad_sequence."""

import torch


def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side="right"):
    n = max(s.shape[0] for s in sequences)
    rows = []
    for s in sequences:
        extra = n - s.shape[0]
        fill = torch.full((extra,) + tuple(s.shape[1:]), padding_value, dtype=s.dtype)
        rows.append(torch.cat([s, fill] if padding_side == "right" else [fill, s]))
    out = torch.stack(rows)
    return out if batch_first else out.transpose(0, 1)
