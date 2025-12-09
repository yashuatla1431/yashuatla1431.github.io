---
layout: post
title: "Rebuilding GPT from Scratch — Kickoff"
date: 2025-11-03
categories: [transformers, gpt, project-log]
excerpt: "Day 1 of my 8-month AI Research Journey: reproducing GPT architecture from memory and understanding attention mechanisms."
---

Today marks the start of my 8-month AI Research Journey.

I completed Andrej Karpathy’s “Let’s Build GPT from Scratch” and began reproducing the model from memory.

### 🧩 Key Takeaways
- The GPT architecture is fundamentally a stack of attention + feedforward blocks.
- Each component (Head, MHA, Block) can be modularized for flexibility.
- The real power lies in understanding the tensor shapes and QKV projections.

### 🚀 Next Steps
- Implement tokenization and training loop.
- Document the math behind attention.
- Commit daily to my [`research-transformers`](https://github.com/yashuatla1431/research-transformers) repo.


