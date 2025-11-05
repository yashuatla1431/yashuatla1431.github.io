# Understanding the Bigram Model: Your First Step Toward Transformers

## What is a Bigram Model?

The bigram model represents the foundational stepping stone toward understanding transformer architecture—the backbone of all modern Large Language Models (LLMs) available today.

As the name suggests, a bigram model considers two units at a time. In our implementation, these units are individual characters: one serves as input, and the other as output. Let's break this down further.

What is a sentence? It's a sequence of words. What are words? They're sequences of letters or characters. The bigram model takes this sequential nature and simplifies it: given one character as input, it predicts what the next character should be.

## How Does the Bigram Model Know What Comes Next?

The answer is simple: **training**. The model learns patterns from data.

I built a bigram model using the Shakespeare dataset and achieved a loss of 2.4, starting from an initial loss of 4.703. Here's an important constraint: **we cannot go below 2.4 loss with a bigram model**. This represents the theoretical lower limit for this architecture.

How can I confidently claim this? The explanation lies in understanding the model's fundamental limitations, which I'll detail at the end of this blog.

## Data Preprocessing: Preparing Shakespeare for Training

Before we can train our model, we need to preprocess the raw Shakespeare text into a format suitable for machine learning.

### Step 1: Reading the Data
```python
f = open('input.txt', 'r')
content = f.read()
```

This is straightforward—we're simply reading the entire Shakespeare text file into memory.

### Step 2: Building the Vocabulary

The vocabulary represents all unique characters present in our dataset. Here's what we get from the Shakespeare text:
```python
['M', 'u', 'o', 'W', 'P', 'R', 'E', 'K', '?', 'h', 'g', 'J', ':', 'c', 'V', ';', 
"'", 'Z', 'b', '-', 'F', 'p', 'j', 'w', 'k', 'G', '&', 'N', 'n', 'C', '.', '\n', 
'q', 'l', 'S', 'i', 'B', 'D', 'r', ' ', 's', 'f', 'd', 'a', ',', 'm', 'Y', '!', 
'x', 'A', 't', 'T', 'X', '$', 'v', 'Q', 'I', 'e', 'y', 'O', 'L', 'H', '3', 'U', 'z']
```

Here's the code that generates this vocabulary and creates encoding/decoding functions:
```python
vocabulary = list(set(content))
n = len(vocabulary)
stoi = {value: key for key, value in enumerate(vocabulary)}
itos = {key: value for key, value in enumerate(vocabulary)}
encode = lambda x: [stoi[i] for i in x]
decode = lambda x: "".join([itos[index] for index in x])
print(vocabulary)
```

**Let's break down each line:**

**`vocabulary = list(set(content))`**
- `set(content)` extracts all unique characters from the text
- `list()` converts the set into a list for easier manipulation

**`n = len(vocabulary)`**
- Stores the total number of unique characters (vocabulary size)

**`stoi = {value: key for key, value in enumerate(vocabulary)}`**
- Creates a "string-to-integer" dictionary
- Maps each character to a unique integer index
- Example: `{'a': 0, 'b': 1, 'c': 2, ...}`

**`itos = {key: value for key, value in enumerate(vocabulary)}`**
- Creates an "integer-to-string" dictionary (reverse mapping)
- Maps each index back to its corresponding character
- Example: `{0: 'a', 1: 'b', 2: 'c', ...}`

**`encode = lambda x: [stoi[i] for i in x]`**
- Lambda function that converts text into a list of integers
- Example: `encode("hello")` → `[7, 4, 11, 11, 14]`

**`decode = lambda x: "".join([itos[index] for index in x])`**
- Lambda function that converts integers back to text
- Example: `decode([7, 4, 11, 11, 14])` → `"hello"`

### Step 3: Setting Global Variables
```python
# Global Variables
batch_size = 4
block_size = 16
e_embd = 32
vocab_size = len(vocabulary)
```

These hyperparameters control our training process:
- **batch_size**: Number of independent sequences processed simultaneously
- **block_size**: Maximum context length (number of characters to consider)
- **e_embd**: Embedding dimension (not used in basic bigram, but included for consistency)
- **vocab_size**: Total number of unique characters

### Step 4: Splitting the Data
```python
# Training and validation data split 
# Train : val : test ==> 80 : 10 : 10
train_records = int(len(content) * 0.8)
val_records = train_records + int(len(content) * 0.1)
train_data = content[:train_records]
val_data = content[train_records:val_records]
test_data = content[val_records:]
print(len(train_data), len(val_data), len(test_data))
```

We split our data into three parts:
- **80% for training**: Used to teach the model patterns
- **10% for validation**: Used to monitor performance during training
- **10% for testing**: Reserved for final evaluation

### Step 5: Creating Training Batches
```python
import torch

def batch(split):
    data = train_data if split == 'train' else val_data
    indexes = torch.randint(0, len(data) - block_size, (batch_size,))
    x_train = torch.stack([torch.tensor(encode(data[index:index+block_size])) 
                           for index in indexes])
    y_train = torch.stack([torch.tensor(encode(data[index+1:index+block_size+1])) 
                           for index in indexes])
    return x_train, y_train
```

**What this function does:**
1. Selects the appropriate dataset (training or validation)
2. Randomly samples `batch_size` starting positions in the text
3. Creates input sequences (`x_train`) of length `block_size`
4. Creates target sequences (`y_train`) offset by one character (the "next" characters)
5. Returns batches as PyTorch tensors

## Building the Bigram Language Model
```python
from torch import nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, x, y=None):
        logits = self.embedding(x)
        if y == None:
            return logits
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(-1))
            return logits, loss
```

**Model Architecture:**
- **Embedding Layer**: A lookup table of size `vocab_size × vocab_size`
- Each character index maps to a vector representing probability distributions for the next character
- The `forward` method computes predictions (logits) and optionally calculates loss

**Testing the Model:**
```python
bigram = BigramLanguageModel()
x_train, y_train = batch('train')
logits, loss = bigram(x_train, y_train)
```

## Validation Function
```python
@torch.no_grad()
def validate(model):
    average_over = 20
    train_loss = 0
    val_loss = 0
    for i in range(average_over):
        x_train, y_train = batch('train')
        x_val, y_val = batch('val')
        train_loss += model(x_train, y_train)[1].item()
        val_loss += model(x_val, y_val)[1].item()
    train_loss /= average_over
    val_loss /= average_over
    print(f"Training Loss: {train_loss} | Validation Loss: {val_loss}")
```

This function:
- Evaluates the model without computing gradients (for efficiency)
- Averages loss over 20 batches for more stable metrics
- Compares training and validation performance to detect overfitting

## Training the Model
```python
import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline

epochs = 10000
l_r = 1e-3
loss_values = []
model = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(), l_r)

for i in range(epochs):
    x_train, y_train = batch('train')
    logits, loss = model(x_train, y_train)
    loss_values.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        validate(model)

validate(model)
```

**Training Loop Breakdown:**
1. **Get a batch** of training data
2. **Forward pass**: Compute predictions and loss
3. **Zero gradients**: Clear previous gradients
4. **Backward pass**: Calculate gradients
5. **Update weights**: Adjust parameters using AdamW optimizer
6. **Validate**: Every 100 epochs, check performance

## Text Generation
```python
@torch.no_grad()
def generate(model, max_output_tokens, token):
    idx = token
    for i in range(max_output_tokens):
        logits = model(idx)
        logits = F.softmax(logits[:, -1, :], dim=-1)
        out = torch.multinomial(logits, num_samples=1)
        idx = torch.cat((idx, out), dim=1)
    return decode(idx[0].tolist())

print(generate(model, 1000, torch.zeros(1, 1).int()))
```

**Generation Process:**
1. Start with an initial token (usually zero/start token)
2. Get predictions for the next character
3. Apply softmax to convert logits to probabilities
4. **Sample** from the probability distribution (introduces randomness)
5. Append the sampled character and repeat

**Why sampling instead of always picking the highest probability?**
If we always chose the most probable character, the model would generate the same text every time, essentially memorizing the training data. Sampling introduces variety and creativity into the output.

## Why 2.4 is the Lower Limit

Here's the mathematical explanation for why a bigram model cannot achieve a loss lower than approximately 2.4 on the Shakespeare dataset.

### The Manual Approach

Imagine creating this model manually:

1. **Build a frequency table** of size `vocab_size × vocab_size`
   - Rows represent the current character
   - Columns represent the next character
   - Each cell counts how many times that character pair appears

   ![Bigram manual](/assets/images/bigram_manual.png)

2. **Convert counts to probabilities**
   - For each row, divide by the row sum
   - Now each row represents a probability distribution (sums to 1.0)

3. **Generate text by sampling**
   - Given input character 'a', look up row 'a'
   - Sample the next character from this probability distribution

### Calculating the Theoretical Loss

The loss function used is **cross-entropy loss**, defined as:
```
Loss = -Σ(y_true × log(y_pred))
```

For a perfect bigram model with the true probability distribution from the data:
```
Loss = -(1/N) × Σ log(P(char_next | char_current))
```

Where:
- N is the total number of character pairs in the dataset
- P(char_next | char_current) is the true probability of the next character given the current one

When you calculate this using the actual bigram frequencies from the Shakespeare dataset, you get approximately **2.4**. This represents the inherent **entropy** or unpredictability in Shakespeare's writing style at the character level.

### Why Can't We Go Lower?

The 2.4 loss represents the fundamental limit because:

1. **Information Bottleneck**: The bigram model only sees one character at a time
2. **No Long-Range Dependencies**: It cannot learn patterns like "if we're in the middle of the word 'beautiful', after 'b-e-a-u-t-i' comes 'f'"
3. **Limited Context**: Language has structure beyond immediate neighbors—grammar, semantics, style
4. **True Data Entropy**: The Shakespeare text itself has this level of randomness when viewed through a single-character window

**The only way to achieve lower loss is to:**
- Use more context (look at multiple previous characters)
- Build more sophisticated architectures (like transformers)
- Or change the dataset entirely

This limitation is precisely why we need more advanced models. The bigram model teaches us the basics, but to truly model language, we must look beyond immediate neighbors and capture longer-range dependencies and patterns.

---

The bigram model, despite its simplicity, demonstrates the core concepts of language modeling: learning patterns from data, predicting what comes next, and understanding the trade-offs between model complexity and performance.