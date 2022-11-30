import os
import urllib.request

import torch
from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import hf_hub_url

WRONG_EMBED_PATH = "wrong.bin"
MIDJOURNEY_EMBED_PATH = "midjourney.bin"


def load_custom_token(
    model_id, use_wrong_token=False, use_midjourney_token=False
) -> tuple[CLIPTextModel, CLIPTokenizer]:
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    if use_wrong_token:
        ensure_token(WRONG_EMBED_PATH, "minimaxir/wrong_embedding_sd_2_0")
        load_learned_embed_in_clip(WRONG_EMBED_PATH, text_encoder, tokenizer)

    if use_midjourney_token:
        ensure_token(MIDJOURNEY_EMBED_PATH, "minimaxir/midjourney_sd_2_0")
        load_learned_embed_in_clip(MIDJOURNEY_EMBED_PATH, text_encoder, tokenizer)

    return text_encoder, tokenizer


def ensure_token(path, repo_id):
    if os.path.exists(path):
        return

    print(f"Downloading token embedding from {repo_id}")
    token_url = hf_hub_url(repo_id=repo_id, filename="learned_embeds.bin")
    with urllib.request.urlopen(token_url) as resp, open(path, "wb") as f:
        f.write(resp.read())


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    # MIT License
    #
    # Copyright (c) 2022 Max Woolf
    #
    # Permission is hereby granted, free of charge, to any person obtaining a
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation
    # the rights to use, copy, modify, merge, publish, distribute, sublicense,
    # and/or sell copies of the Software, and to permit persons to whom the
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.

    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    print("Trained token", trained_token)
