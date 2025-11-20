#!/usr/bin/env python3

import io
import math
import requests
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import clip
from torchvision import datasets

IMAGE_URLS = [
    "https://images.pexels.com/photos/36744/agriculture-arable-clouds-countryside.jpg",
    "https://images.pexels.com/photos/825947/pexels-photo-825947.jpeg",
    "https://images.pexels.com/photos/34044163/pexels-photo-34044163.jpeg",
    "https://live.staticflickr.com/840/43380549381_004601c7ac_h.jpg",
    "https://live.staticflickr.com/2404/2020522557_d1aa0a1066_k.jpg",
]

# 10,000 most common English words, from https://github.com/first20hours/google-10000-english
WORDLIST_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"

MIN_WORDS = 10_000
MAX_WORDS = 100_000

TOP_K = 8 # how many top matches we look at
MAX_CAPTIONS = 10_000 # how many random captions to sample per image when searching
MAX_CAPTION_WORDS = 4 # max number of words per random caption. CLIP seems to like it when this value is smaller, definitely in the single digits


def normalise_features(features):
    # cosine similarity reduces to a dot product on unit-length vectors
    norm = features.norm(dim=-1, keepdim=True)
    return features / norm


def encode_texts_in_batches(model, texts, device, batch_size=256):
    # using CLIP in small batches helps avoid running out of GPU memory
    all_features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="encoding texts", leave=False):
            batch_texts = texts[i : i + batch_size]
            tokens = clip.tokenize(batch_texts).to(device)
            text_features = model.encode_text(tokens)
            text_features = normalise_features(text_features)
            all_features.append(text_features)
    return torch.cat(all_features, dim=0)


def find_top_k_words_for_each_image(image_features, word_features, words, top_k):
    similarities = image_features @ word_features.T  # [N, M]
    top_scores, top_indices = similarities.topk(k=top_k, dim=1)

    top_words_per_image = []
    top_scores_per_image = []

    for row_scores, row_indices in zip(top_scores, top_indices):
        indices = row_indices.tolist()
        top_words_per_image.append([words[i] for i in indices])
        top_scores_per_image.append(row_scores.tolist())

    return top_words_per_image, top_scores_per_image


def make_prompt_list_from_words(words, template="A photo of a {}"):
    return [template.format(w) for w in words]


def find_best_free_caption_for_images(model, image_features, device, top_words_per_image):
    # for each image, we (1) take the top K single words, (2) generate random captions from those words, (3) score them with CLIP, and (4) keep the best one
    num_images = image_features.shape[0]
    captions = []
    scores = []

    for i in tqdm(range(num_images), desc="searching free captions"):
        words = top_words_per_image[i]

        candidate_captions = []
        # not sure what the reasoning would be for adding or removing prefixes, but there seems to be worse performance when removing ALL of them. I think the prefixes make sense, if the captions are crowd-sourced and human-written
        prefixes = ["", "a photo of", "a close-up of", "a scenic view of"]

        # randomly sample captions
        for _ in range(MAX_CAPTIONS):
            length = random.randint(2, min(MAX_CAPTION_WORDS, len(words)))
            chosen_words = random.sample(words, length)
            prefix = random.choice(prefixes)

            if prefix:
                caption = prefix + " " + " ".join(chosen_words)
            else:
                caption = " ".join(chosen_words)

            candidate_captions.append(caption)

        candidate_features = encode_texts_in_batches(model, candidate_captions, device)

        # similarity between this image and all candidates
        sims = image_features[i].unsqueeze(0) @ candidate_features.T  # [1, num_captions]
        best_score, best_idx = sims.squeeze(0).max(dim=0)

        captions.append(candidate_captions[best_idx.item()])
        scores.append(best_score.item())

    return captions, scores


def main():
    # works on my machine :)
    assert torch.cuda.is_available()
    device = "cuda"

    print("loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    print("loading images...")

    image_tensors = []
    for url in IMAGE_URLS:
        print(f"downloading image: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensors.append(image_tensor)
    assert len(image_tensors) > 0
    image_batch = torch.cat(image_tensors, dim=0)
    print(f"loaded {image_batch.shape[0]} images")

    url = WORDLIST_URL
    print(f"downloading word list from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    words = []
    seen = set()
    for line in lines:
        word = line.strip().lower()
        if not word:
            continue
        if not word.isalpha():
            continue
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= MAX_WORDS:
            break
    assert len(words) >= MIN_WORDS, f"word list too small: have {len(words)}, need at least {MIN_WORDS}"
    print(f"loaded {len(words)} words")
    word_list = words

    print("encoding images...")
    with torch.no_grad():
        image_features = model.encode_image(image_batch.to(device))
    image_features = normalise_features(image_features)

    print("encoding single words as text...")
    word_text_features = encode_texts_in_batches(model, word_list, device)

    #
    # Part 1.1: `For each image I, Find the single word W that maximizes the cosine similarity (CLIP(I), CLIP(W)).`
    #

    print(f'finding top {TOP_K} single words W for each image (max CLIP(I), CLIP(W))...')
    top_words_plain, top_scores_plain = find_top_k_words_for_each_image(
        image_features,
        word_text_features,
        word_list,
        TOP_K,
    )

    print('encoding "A photo of a W" prompts...')
    prompt_list = make_prompt_list_from_words(word_list)  # prompts for ALL words
    prompt_features = encode_texts_in_batches(model, prompt_list, device)

    #
    # Part 1.2: `For each image I, find the "simple structured caption" with 1 word so that maximizes the cosine similarity (CLIP(I), CLIP("A photo of a W")).`
    #

    print(f'finding top {TOP_K} "A photo of a W" captions for each image...')
    top_words_prompt, top_scores_prompt = find_top_k_words_for_each_image(
        image_features,
        prompt_features,
        word_list,
        TOP_K,
    )

    #
    # Part 1.3: `For each image I, find the arbitrary caption C that maximizes the cosine similarity (CLIP(I), CLIP(C)).`
    #

    print(f"finding best free caption using up to {MAX_CAPTIONS} random captions per image...")
    free_captions, free_caption_scores = find_best_free_caption_for_images(
        model,
        image_features,
        device,
        top_words_plain,  # use the top-K plain words as vocab per image
    )

    #
    # Part 2: `Find the pair of ("regular image",caption) (e.g. not an image of all white pixels or other super wierd artificial image), that gives the largest cosine similarity you can.`
    #
    # This is similar to Part 1.3, except we need to find any images. I decided to sample 2000 images randomly from CIFAR-10 (any more and I run out of GPU memory)

    NUM_DATASET_IMAGES = 2000
    NUM_GLOBAL_CAPTIONS = MAX_CAPTIONS

    print(f"\n[Part 2] loading CIFAR-10 and sampling {NUM_DATASET_IMAGES} images...")
    cifar10 = datasets.CIFAR10(root="data/cifar10", train=True, download=True)
    total = len(cifar10)
    if NUM_DATASET_IMAGES > total:
        NUM_DATASET_IMAGES = total

    # random subset of CIFAR-10 indices
    subset_indices = torch.randperm(total)[:NUM_DATASET_IMAGES]

    # encode sampled images with CLIP
    dataset_image_tensors = []
    dataset_labels = []
    for index in tqdm(subset_indices, desc="Part 2: loading CIFAR images"):
        image, label = cifar10[index.item()]
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        dataset_image_tensors.append(img_tensor)
        dataset_labels.append(label)

    dataset_image_batch = torch.cat(dataset_image_tensors, dim=0)

    print(f"[Part 2] encoding {NUM_DATASET_IMAGES} CIFAR-10 images with CLIP...")
    with torch.no_grad():
        dataset_image_features = model.encode_image(dataset_image_batch.to(device))
    dataset_image_features = normalise_features(dataset_image_features)

    # build a global pool of random captions from the 10k-word list
    print(f"[Part 2] generating {NUM_GLOBAL_CAPTIONS} random captions...")
    global_captions = []
    prefixes = ["", "a photo of", "a close-up of", "a scenic view of"]

    for _ in tqdm(range(NUM_GLOBAL_CAPTIONS), desc="Part 2: generating captions"):
        length = random.randint(2, MAX_CAPTION_WORDS)
        length = min(length, len(word_list))
        chosen_words = random.sample(word_list, length)
        prefix = random.choice(prefixes)

        if prefix:
            caption = prefix + " " + " ".join(chosen_words)
        else:
            caption = " ".join(chosen_words)

        global_captions.append(caption)

    # I encode the random captions once to reuse for all CIFAR images, because it'd be too expensive to generate unique ones for each image
    print("[Part 2] encoding random captions with CLIP...")
    global_caption_features = encode_texts_in_batches(model, global_captions, device)

    # again, cosine similarity = dot product because everything is normalised
    print("[Part 2] computing global similarity matrix...")
    sims = dataset_image_features @ global_caption_features.T  # [N_images, N_captions]

    flat_sims = sims.view(-1)
    top_values, top_indices = flat_sims.topk(TOP_K)
    n_captions = global_caption_features.shape[0]

    print("\n[Part 2] top image–caption pairs:")

    for rank in tqdm(range(TOP_K), desc="Part 2: saving top-K pairs"):
        flat_index = top_indices[rank].item()
        score = top_values[rank].item()

        image_index = flat_index // n_captions
        caption_index = flat_index % n_captions

        caption = global_captions[caption_index]
        dataset_index = subset_indices[image_index].item()
        label = dataset_labels[image_index]
        label_name = cifar10.classes[label]

        print(f"\tRank {rank+1}:")
        print(f"\t\tCIFAR-10 dataset index: {dataset_index}")
        print(f"\t\tCIFAR-10 class label:   {label_name} (id={label})")
        print(f"\t\tCaption:                \"{caption}\"")
        print(f"\t\tCosine similarity:      {score:.4f}")

        # save the visualisation for this pair
        image, _ = cifar10[dataset_index]  # PIL Image, 32x32

        base_width, base_height = image.size
        scale = 8
        image = image.resize((base_width * scale, base_height * scale), Image.NEAREST)
        upscaled_width, upscaled_height = image.size

        caption_to_draw = caption
        max_characters = 100
        if len(caption_to_draw) > max_characters:
            caption_to_draw = caption_to_draw[:max_characters] + "..."

        text_band = 60
        out_image = Image.new("RGB", (upscaled_width, upscaled_height + text_band), color=(255, 255, 255))
        out_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(out_image)
        text_x = 5
        text_y = upscaled_height + 5
        draw.text((text_x, text_y), caption_to_draw, fill=(0, 0, 0))

        out_path = f"top_{rank}_free_pair.png"
        out_image.save(out_path)
        print(f"    Saved visualisation to {out_path}\n")

    # final report
    print("\n=== results ===\n")
    for index, url in enumerate(IMAGE_URLS):
        print(f"image {index}: {url}")

        print(f"\tTop {TOP_K} single words W (CLIP(I), CLIP(W)):")
        for rank in range(TOP_K):
            w = top_words_plain[index][rank]
            s = top_scores_plain[index][rank]
            print(f"\t  {rank+1:2d}. {w:>15}  (score={s:.4f})")

        print(f'\n\tTop {TOP_K} simple captions "A photo of a W":')
        for rank in range(TOP_K):
            w = top_words_prompt[index][rank]
            s = top_scores_prompt[index][rank]
            print(f'\t  {rank+1:2d}. "A photo of a {w}"  (score={s:.4f})')

        print(
            f'\n\tBest free caption (from {MAX_CAPTIONS} random candidates): '
            f'"{free_captions[index]}" (score={free_caption_scores[index]:.4f})'
        )
        print("")


main()
