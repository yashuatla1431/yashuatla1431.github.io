---
layout: post
title: "Tokenization: Teaching Computers to Read (Sort Of)"
date: 2026-02-26
categories: [fundamentals, nlp, hands-on]
excerpt: "Building a BPE tokenizer from scratch. Turns out, computers don't actually understand words - they just pretend really well."
---

# Why Do We Even Need Tokenization?

Here's the thing: computers are really, REALLY good at working with numbers. Like, scary good. But text? Words? Sentences? Nah, they don't get it at all.

So when we want to build language models (like GPT, BERT, or whatever the cool kids are using these days), we need to convert text into numbers somehow. That's where tokenization comes in.

You might think, "Easy! Just give each letter a number, right? A=1, B=2, C=3..." And yeah, you COULD do that. But that's super inefficient. Why?

Because then common words like "the" become three separate tokens [t, h, e], and the model has to learn that these three always appear together. It's like learning a language by memorizing individual letters instead of words. Painful and slow.

## The Smart Way: Byte Pair Encoding (BPE)

BPE is this clever algorithm that sits somewhere between character-level and word-level tokenization. It's like Goldilocks - not too granular, not too coarse, just right.

Here's the genius part: BPE starts with individual characters (or bytes, to be precise) and then **merges the most frequent pairs** repeatedly until you hit your target vocabulary size.

Let's say you have the text: "hello hello hello world"

- At first, everything is individual characters: `h e l l o   w o r l d`
- BPE notices that `l` and `l` appear together a LOT (in "hello")
- So it merges them: `h e ll o   w o r l d`
- Then it notices `e` and `ll` appear together often
- Merge again: `h ell o   w o r l d`
- And so on...

Eventually, "hello" might become a single token because it appears so frequently. But rare words? They stay split up into smaller pieces. Smart, right?

## Why Bytes and Not Characters?

Okay, quick Unicode rant because this is actually important.

If you've ever dealt with text from different languages, emojis, or special symbols, you know it's a nightmare. There are like a million different characters in Unicode (okay, not a million, but close enough).

Here's where it gets interesting. When you encode text to UTF-8 bytes, ANY character - English letters, emojis, Chinese characters, whatever - becomes a sequence of bytes (numbers from 0-255). So your base vocabulary is just 256 tokens. Nice and fixed.

Let me show you:

```python
text = "hello☆"
bytes_list = list(text.encode('utf-8'))
print(bytes_list)
# Output: [104, 101, 108, 108, 111, 226, 152, 134]
```

See that? "hello" is straightforward: [104, 101, 108, 108, 111]. But the star emoji ☆? It's three bytes: [226, 152, 134]. UTF-8 handles it automatically.

This is why modern tokenizers (like the one in GPT) work at the byte level. You can handle ANY text thrown at you without manually maintaining a list of every possible character.

## Building Our Mini BPE Tokenizer

Alright, enough theory. Let's build this thing from scratch and see how it actually works.

### Step 1: Getting Pair Frequencies

First, we need a way to count how often each pair of tokens appears next to each other.

```python
def get_stats(tokens):
    freq_dict = {}
    for i, j in zip(tokens[:-1], tokens[1:]):
        freq_dict[(i,j)] = freq_dict.get((i,j), 0) + 1
    return freq_dict
```

**What's this doing?**

We're zipping the tokens with themselves, shifted by one position. So if you have `[1, 2, 3, 4]`, you get pairs `[(1,2), (2,3), (3,4)]`. Then we just count how many times each pair appears.

Super simple, but this is the heart of BPE. We need to know which pairs are the most common so we can merge them.

### Step 2: Merging Pairs

Once we know the most frequent pair, we need to merge it into a single token.

```python
def merge(tokens, pair, new_token_id):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            merged_tokens.append(new_token_id)
            i += 2  # Skip both tokens
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens
```

**The logic here:**

We scan through the token list. Whenever we find our target pair, we replace it with a new token ID and skip ahead by 2. Otherwise, we just copy the token as-is.

For example, if we're merging the pair `(104, 101)` (which is "he" in bytes) into token `256`, then:
- Input: `[104, 101, 108, 108, 111]` (hello)
- Output: `[256, 108, 108, 111]` (merged "he" + llo)

### Step 3: Training the Tokenizer

Now we put it all together. Training a BPE tokenizer means repeatedly finding the most common pair and merging it until we reach our desired vocabulary size.

```python
def train(text, vocab_size, exp_vocab_size):
    tokens = list(text.encode('utf-8'))
    merged_tokens = {}

    while vocab_size < exp_vocab_size:
        # Find most frequent pair
        dic = get_stats(tokens)
        max_value = max(dic.items(), key=lambda x: x[1])

        # Assign it a new token ID
        vocab_size += 1
        merged_tokens[max_value[0]] = vocab_size

        # Merge that pair in our token list
        tokens = merge(tokens, max_value[0], vocab_size)

    return tokens, merged_tokens
```

**Breaking it down:**

1. Start by converting the text to UTF-8 bytes (our base tokens, 0-255)
2. Count all pair frequencies
3. Find the pair that appears most often
4. Create a new token ID for that pair (starting from 256 and going up)
5. Merge all occurrences of that pair in the token list
6. Repeat until we hit the vocabulary size we want

The `merged_tokens` dictionary keeps track of all the merges we made. It maps `(token1, token2) -> new_token_id`. We'll need this for encoding and decoding later.

### Step 4: Encoding New Text

Once we've trained the tokenizer, we can use it to encode any new text:

```python
def encode(text, merged_tokens):
    tokens = list(text.encode('utf-8'))  # Start with UTF-8 bytes

    # Sort merges by token ID to apply in order
    merges = sorted(merged_tokens.items(), key=lambda x: x[1])

    for pair, new_id in merges:
        tokens = merge(tokens, pair, new_id)

    return tokens
```

**Why sort by token ID?**

Because we need to apply the merges in the SAME ORDER we learned them during training. If we learned to merge "th" before "the", we need to apply "th" first, then "the". Otherwise, the tokens won't match what the model expects.

It's like following a recipe - you can't just do the steps in random order and expect it to work.

### Step 5: Decoding Tokens Back to Text

This is where it gets a bit tricky. We need to reverse all those merges to get back to the original UTF-8 bytes, then decode to text.

```python
def decode(token_ids, merged_tokens):
    vocab_map = {value: key for key, value in merged_tokens.items()}
    data = []

    def unwrap(token_id):
        pair = vocab_map[token_id]
        tokens = []

        # Base case: both tokens in the pair are original bytes
        if pair[0] not in vocab_map and pair[1] not in vocab_map:
            return list(pair)

        # Recursively unwrap the first token
        if pair[0] in vocab_map:
            tokens.extend(unwrap(pair[0]))
        else:
            tokens.append(pair[0])

        # Recursively unwrap the second token
        if pair[1] in vocab_map:
            tokens.extend(unwrap(pair[1]))
        else:
            tokens.append(pair[1])

        return tokens

    for token in token_ids:
        if token in vocab_map:
            data.extend(unwrap(token))
        else:
            data.append(token)

    return bytes(data).decode('utf-8')
```

**What's happening here?**

Think of this as unwrapping a gift recursively. Each merged token is like a wrapped box containing two smaller tokens (which might themselves be wrapped boxes).

For example, if token `258` was created by merging tokens `256` and `257`, and token `256` was created by merging bytes `104` and `101`, then:
- `unwrap(258)` → `unwrap(256)` + `unwrap(257)`
- `unwrap(256)` → `[104, 101]`
- And so on...

Eventually, we get back to the original UTF-8 bytes, which we can decode to text.

## Let's Test This Thing

Time to see if our tokenizer actually works. Let's train it on some text and then encode/decode a sentence:

```python
# Training text
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear..."

# Convert to bytes
data = list(text.encode('utf-8'))
vocab_size = max(data)  # Base vocabulary: 0-255
exp_vocab_size = vocab_size + 30  # Add 30 merge tokens

# Train the tokenizer
tokens, merged_tokens = train(text, vocab_size, exp_vocab_size)

# Test it on new text
sample = "Hey i am yashwanth happy to meet you all how are you"
original_length = len(list(sample.encode('utf-8')))
print(f"Original token count: {original_length}")

# Encode
tokens_encoded = encode(sample, merged_tokens)
print(f"After BPE: {len(tokens_encoded)} tokens")

# Decode
text_decoded = decode(tokens_encoded, merged_tokens)
print(f"Decoded text: {text_decoded}")
```

**Output:**
```
Original token count: 52
After BPE: 47 tokens
Decoded text: Hey i am yashwanth happy to meet you all how are you
```

Look at that! We compressed 52 tokens down to 47 by merging common patterns. And we can perfectly decode it back. Not bad for a few lines of code!

The compression isn't huge because we only trained on a small text with 30 merges. In real tokenizers (like GPT's), they train on massive datasets with vocabularies of 50,000+ tokens, which gets way better compression.

## Putting It All Together: The BPETokenizer Class

Now let's wrap everything into a clean class so we can reuse it easily:

```python
class BPETokenizer:
    def __init__(self):
        self.merged_tokens = {}
        self.vocab_size = None

    def get_stats(self, tokens):
        freq_dict = {}
        for i, j in zip(tokens[:-1], tokens[1:]):
            freq_dict[(i,j)] = freq_dict.get((i,j), 0) + 1
        return freq_dict

    def merge(self, tokens, pair, new_token_id):
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged_tokens.append(new_token_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def train(self, text, vocab_size, exp_vocab_size):
        tokens = list(text.encode('utf-8'))
        self.merged_tokens = {}

        while vocab_size < exp_vocab_size:
            dic = self.get_stats(tokens)
            max_value = max(dic.items(), key=lambda x: x[1])
            vocab_size += 1
            self.merged_tokens[max_value[0]] = vocab_size
            tokens = self.merge(tokens, max_value[0], vocab_size)

        self.vocab_size = vocab_size
        return tokens, self.merged_tokens

    def encode(self, text, merged_tokens):
        tokens = list(text.encode('utf-8'))
        merges = sorted(merged_tokens.items(), key=lambda x: x[1])

        for pair, new_id in merges:
            tokens = self.merge(tokens, pair, new_id)

        return tokens

    def decode(self, token_ids, merged_tokens):
        vocab_map = {value: key for key, value in merged_tokens.items()}
        data = []

        def unwrap(token_id):
            pair = vocab_map[token_id]
            tokens = []
            if pair[0] not in vocab_map and pair[1] not in vocab_map:
                return list(pair)
            if pair[0] in vocab_map:
                tokens.extend(unwrap(pair[0]))
            else:
                tokens.append(pair[0])
            if pair[1] in vocab_map:
                tokens.extend(unwrap(pair[1]))
            else:
                tokens.append(pair[1])
            return tokens

        for token in token_ids:
            if token in vocab_map:
                data.extend(unwrap(token))
            else:
                data.append(token)

        return bytes(data).decode('utf-8')
```

Now you can just instantiate it and use it like:

```python
tokenizer = BPETokenizer()
tokens, merged_tokens = tokenizer.train(text, vocab_size, exp_vocab_size)
encoded = tokenizer.encode("some new text", merged_tokens)
decoded = tokenizer.decode(encoded, merged_tokens)
```

Clean and reusable. That's how we like it.

## Why Does This Matter?

You might be thinking, "Okay cool, but why should I care about tokenization? The model does the hard work anyway, right?"

Well, yes, but also no. Tokenization is actually a HUGE deal in modern NLP. Here's why:

**1. Vocabulary size affects everything**

If your vocabulary is too small, every sentence becomes a million tokens, which makes training slow and expensive. If it's too large, the model has to learn representations for tokens it rarely sees, which is wasteful.

BPE finds a sweet spot automatically based on your training data.

**2. It handles rare words gracefully**

With word-level tokenization, every rare word becomes an `<UNK>` (unknown) token, and the model learns nothing. With BPE, rare words get split into subword pieces, and the model can still make sense of them.

For example, if the model has seen "play", "ing", and "ground", it can probably figure out "playground" and "playing" even if it's never seen those exact words.

**3. It's multilingual-friendly**

Because BPE works at the byte level, it can handle ANY language without special casing. English, Chinese, Arabic, emoji spam - doesn't matter. It all becomes bytes, and BPE merges the common patterns.

This is why GPT can do reasonably well in multiple languages with a single tokenizer.

## The Limitations (Because Nothing Is Perfect)

Okay, real talk: BPE isn't perfect. Here are some quirks:

**1. Encoding is slow**

Notice how we have to apply merges sequentially in order? That's O(n × m) where n is text length and m is number of merges. For a 50,000-token vocabulary, that adds up.

Real implementations (like tiktoken) use optimized data structures and caching to speed this up, but it's still a bottleneck.

**2. It's greedy**

BPE always merges the most frequent pair at each step. But this might not be globally optimal. There's a chance a different sequence of merges would result in better compression or more meaningful tokens.

But hey, it's fast and works well in practice, so we use it anyway.

**3. Tokenization isn't perfect**

Sometimes BPE creates weird tokens. For example, " the" (with a leading space) might be a single token, but "the" (without a space) is split differently. This can confuse models, especially at the start of sentences.

Also, trailing spaces can affect tokenization, which is why you sometimes see language models generate weird spacing.

## Wrapping Up

So there you have it: Byte Pair Encoding from scratch. We built a tokenizer that:
- Converts any text to UTF-8 bytes
- Learns to merge frequent pairs
- Compresses text into fewer tokens
- Can encode and decode losslessly

This is the same basic algorithm used by GPT, BERT, and most modern language models. Sure, they have fancy optimizations and train on way more data, but the core idea is exactly what we just implemented.

Next time someone mentions "tokenization" in an ML paper, you can nod knowingly because you've built one yourself. Not bad for a day's work!

Now go forth and tokenize all the things. Or don't. I'm not your boss.

---

**P.S.** If you're wondering why your language model sometimes acts weird with punctuation or spacing, now you know - blame the tokenizer.
