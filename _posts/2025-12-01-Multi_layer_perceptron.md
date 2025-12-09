---
layout: post
title: "Multi-Layer Perceptron: When One Layer Just Isn't Enough"
date: 2025-12-01
categories: [fundamentals, language-models, hands-on]
excerpt: "Building a multi-layer perceptron to predict names character by character. Way better than bigram, but still not transformer-level magic."
---

# What Even Is a Multi-Layer Perceptron?

So basically, let's talk about Multi-Layer Perceptron (MLP). First, what is a perceptron? A perceptron is nothing but a neuron, which I already explained in detail in my [neuron blog](link).

The terminology here is pretty straightforward. **Perceptron** + **Layer** = we're putting multiple perceptrons in a layer. Then **Multi-Layer** = we're stacking multiple layers between input and output. This whole thing? That's just a neural network. If you have a TON of layers (like, a lot), then it becomes a **deep neural network**. Fancy name, same concept.

If you haven't read my neuron blog, go check out the neural network section there first. But here's the quick version: the first layer is technically the input layer (even though logically it doesn't really count as part of the network, but we'll call it a layer anyway). Then there's the output layer at the end.

## How Many Output Neurons Do We Need?

The number of neurons in the output layer depends on what answer we're looking for. Let's say we're doing image classification - is this a cat or not? Then the final layer will have just **1 neuron** that outputs 0 for "not a cat" and 1 for "cat".

But what if we're trying to detect which number is shown in an image (0 to 9)? Then the output layer needs **10 neurons**. If the image shows a 2, only the 3rd neuron (counting from 0) fires with a 1, and the rest stay at 0. Simple, right?

## Quick Detour: Images as Inputs

Now, how do we even give an image as input to a neural network?

Let me use the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) as an example because it's kinda interesting (and also everywhere in ML tutorials, haha). The MNIST dataset has handwritten digits from 0-9 in black and white.

Normally, for images, you'd use something called a **CNN (Convolutional Neural Network)** for better spatial recognition and pixel relationship detection. But as beginners, we just use a regular deep neural network with MNIST to understand the basics.

Here's the trick: the image is basically a matrix where each pixel is either 0 or 1 (black or white). We convert this matrix into a long list and feed it to the first layer.

![MNIST Input Example](/assets/images/mnist.jpg)

We need to give the network **only numbers**, not words or other symbols. Why? Because computers are specialized in doing circus tricks with numbers, not English.

## Alright, Let's Build This Thing

Now, without further delay, let's jump into some code and build our MLP to predict the next character in a name.

If you've seen my bigram model blog, you know we used the **previous character** as context to predict the next one. But that's not great because you're only looking at one character back - your accuracy is going to be terrible.

So let's level up. What if we give the model **more context**? Let's say 3 characters. We call this the **window size** in MLP terminology.

### Loading the Dataset

First, let's load our dataset, which is just a bunch of names:

```
yashwanth
monkey
donkey
...
```

We're loading over 100,000 names here. Now let's do some preprocessing:

```python
f = open('names.txt','r')
names = f.read().split('\n')
name_string = '.'.join(names)
distinct_chars = sorted(list(set([char for char in name_string])))
stoi = {value:key for key,value in enumerate(distinct_chars)}
itos = {key:value for key,value in enumerate(distinct_chars)}
m = len(distinct_chars)
```

**What's happening here?**

In the first 3 lines, we're loading the dataset and creating one big string with all the names, separated by periods (`.`). Then we're getting all the **distinct characters** in that big string.

Next, we create two dictionaries:
- **stoi** (string to integer): maps each character to an index (0-26)
- **itos** (integer to string): the reverse mapping

They look like this:
```python
{'.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, ..., 'z': 26}
{0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', ..., 26: 'z'}
```

You might be thinking, "Am I dumb or what? Why create the big string just for this?" Well, if the dataset changes and has all kinds of ASCII characters (like emojis or special symbols), this approach handles it automatically without you manually listing everything. Smart, right?

### Building the Dataset

```python
# building the dataset
window = 3
X = []
Y = []
for word in names:
    total_word = '...'+word+'.'
    start = 0
    end = start + window
    while end < len(total_word):
        x = total_word[start:end]
        y = total_word[end]
        start+=1
        end+=1
        X.append([stoi[char] for char in x])
        Y.append(stoi[y])
X = torch.tensor(X)
Y = torch.tensor(Y)

# let's split up the dataset
# 80-10-10
numtr = int(X.size()[0]*(80/100))
numdev = numtr + int(X.size()[0]*(10/100))
Xtr,Ytr = X[:numtr],Y[:numtr]
Xdev,Ydev = X[numtr:numdev],Y[numtr:numdev]
Xtest,Ytest = X[numdev:],Y[numdev:]
print(Xtr.size(),Xdev.size(),Xtest.size())
```

**What's this doing?**

We're creating training examples using a sliding window. For each name, we add `...` at the start and `.` at the end (the dots act as start/stop markers). Then we slide a window of size 3 across the name.

For example, with the name "yashwanth":
- Input: `...` → Output: `y`
- Input: `..y` → Output: `a`
- Input: `.ya` → Output: `s`
- And so on...

We convert everything to numbers using our `stoi` dictionary and store them in `X` (inputs) and `Y` (outputs).

Finally, we split the data: **80% for training**, **10% for validation (dev set)**, and **10% for testing**. This is standard practice so we can check if our model is actually learning or just memorizing.

### Building the Neural Network

```python
# building the neural network
layer2_size = 200
num_features = 10
g = torch.Generator().manual_seed(33321)
C = torch.randn((m,num_features),generator=g)
W1 = torch.randn((num_features*window,layer2_size),generator=g) * ((5/3)/((num_features*window)**0.5))
W2 = torch.randn((layer2_size,m),generator=g) * 0.01
b2 = torch.zeros((1,m))
bngain = torch.ones((1,layer2_size))
bnbias = torch.zeros((1,layer2_size))
bnmean_running = torch.zeros((1,layer2_size))
bnstd_running = torch.ones((1,layer2_size))
parameters = [C,W1,W2,b2,bngain,bnbias]
for param in parameters:
    param.requires_grad = True
```

**Okay, what's all this mess?**

We're setting up the layers of our neural network. Don't worry about every single variable - here's the big picture:

- **C**: This is our character embedding matrix. It converts each character (0-26) into a vector of 10 features. Think of it like giving each character a personality with 10 traits.
- **W1**: Weights for the first layer. We take 3 characters (window size), each with 10 features, and multiply them to get 200 neurons in the hidden layer.
- **W2**: Weights for the second layer. Takes the 200 neurons and projects them back to 27 outputs (one for each character).
- **b2**: Bias for the output layer (helps shift the predictions).
- **bngain, bnbias**: These are for batch normalization (keeps the neuron activations from going crazy during training).
- **bnmean_running, bnstd_running**: Running averages for batch normalization during inference.

We mark all these as `requires_grad = True` so PyTorch knows to update them during training.

### Training the Model

```python
# training loop
import torch.nn.functional as F
import math
epochs = 20000
lr = 0.001
lri = [0,-1,-2,-3,-4]
lre = [10**z for z in lri]
lrei = 0
epochs_graph = []
loss_graph = []
epselon = 10**-3

for i in range(epochs):
    # mini batch
    mini_batch = torch.randint(low=0,high=Xtr.size()[0],size=(32,))
    x = Xtr[mini_batch]

    # linear layer
    h = C[x].view(-1,num_features*window)@W1

    # batch norm layer
    batch_mean = h.mean(dim=0,keepdim=True)
    batch_std = h.std(dim=0,keepdim=True)
    h = bngain*((h - batch_mean)/(batch_std + epselon)**0.5) + bnbias
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * batch_mean
        bnstd_running = 0.999 * bnstd_running + 0.001 * batch_std

    # tanh activation
    a1 = torch.tanh(h)

    # calculate loss
    loss = F.cross_entropy(a1@W2 + b2,Ytr[mini_batch])

    # reset gradients
    for param in parameters:
        param.grad = None

    # backpropagation
    loss.backward()

    # update learning rate schedule
    if i%(math.ceil(epochs//4)) == 0:
        lrei+=1

    # update weights
    for param in parameters:
        param.data -= lre[lrei]*param.grad

    if i%1000 == 0:
        print(f"EPOCH : {i} -> LOSS : {loss}")
        epochs_graph.append(i)
        loss_graph.append(loss.data.item())

final_train_set_loss = loss
print(f"FINAL LOSS : {final_train_set_loss}")
```

**The training loop explained (without going insane):**

We're running 20,000 epochs (iterations). For each epoch:

1. **Mini-batch**: Randomly pick 32 examples from the training set. Why? Because training on the full dataset every time is slow and unnecessary.

2. **Forward pass**:
   - Look up character embeddings using `C`
   - Multiply by weights `W1` to get the hidden layer `h`
   - Apply batch normalization (keeps values stable)
   - Apply tanh activation (squashes values between -1 and 1)
   - Multiply by `W2` and add `b2` to get predictions

3. **Calculate loss**: How wrong are our predictions? Cross-entropy loss tells us.

4. **Backward pass**: PyTorch calculates gradients (how to adjust weights to improve).

5. **Update weights**: Subtract a small amount (learning rate × gradient) from each parameter.

6. **Learning rate schedule**: Every 5,000 epochs, we reduce the learning rate so the model takes smaller, more careful steps as it gets closer to the solution.

### Visualizing Layer Activations

```python
import matplotlib.pyplot as plt
fig,axes = plt.subplots(1,2,figsize=(10,4))
axes[0].hist(h.view(-1).detach().numpy(),bins=100,color='skyblue')
axes[0].set_title("Layer 1 Before Activation")
axes[1].hist(a1.view(-1).detach().numpy(),bins=100,color='skyblue')
axes[1].set_title("Layer 1 After Activation")
plt.show()
```

We're plotting histograms to see how the neuron values are distributed before and after the tanh activation. This helps us understand if neurons are "saturating" (getting stuck at extreme values) or if they're nicely distributed.

![Layer Activation Distribution](/assets/images/layer_act.png)

See how the tanh activation squashes everything between -1 and 1? That's exactly what we want.

### Checking Performance on Dev Set

```python
# calculating dev set loss
@torch.no_grad()
def calculate_loss():
    random_mini_batch = torch.randint(low=0,high=Xdev.size()[0],size=(32,))
    x = Xdev[random_mini_batch]
    h = C[x].view(-1,num_features*window)@W1
    h = bngain * ((h - bnmean_running)/(bnstd_running + epselon)**0.5) + bnbias
    a1 = torch.tanh(h)
    loss = F.cross_entropy(a1@W2 + b2,Ydev[random_mini_batch])
    return loss

print(f"Dev Set Loss: {calculate_loss()} | Train Set Loss: {final_train_set_loss}")
```

**Why do we need this?**

During training, we only looked at the training set. But we need to check: did the model actually **learn** or did it just **memorize**?

The dev set (validation set) is data the model has never seen before. If the dev loss is close to the train loss, great! If it's way higher, we're overfitting (memorizing instead of learning).

### Generating New Names

```python
# inference
start = [0,0,0]
@torch.no_grad()
def inference(start):
    inference_window = start
    result = []
    while True:
        x = C[inference_window]
        h = x.view(-1,num_features*window)@W1
        h = bngain * ((h - bnmean_running)/(bnstd_running + epselon)**0.5) + bnbias
        a1 = torch.tanh(h)
        logits = a1@W2 + b2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1).item()
        inference_window = inference_window[1:] + [ix]
        result.append(ix)
        if ix == 0:
            break
    return ''.join([itos[index] for index in result])

num_of_words = 20
for i in range(num_of_words):
    print(f"Word {i}: {inference(start)}")
```

**The fun part - generating new names!**

We start with three dots (`...`), which is `[0, 0, 0]` in our encoding. Then:

1. Pass the current window through the network
2. Get probabilities for the next character
3. **Sample** a character (not always the highest probability - that keeps it interesting)
4. Slide the window: drop the first character, add the new one
5. Keep going until we generate a `.` (end marker)

This generates completely new names that the model has never seen before. Some will be realistic-sounding, some will be... creative. That's the beauty of it.

**Here's what our model generated:**

```
word 0 : ahmeyshil.
word 1 : khasleeshlangerranz.
word 2 : belo.
word 3 : marie.
word 4 : msydenna.
word 5 : yza.
word 6 : risna.
word 7 : shriswrytia.
word 8 : bavolah.
word 9 : pen.
word 10 : ariy.
word 11 : pannessa.
word 12 : mari.
word 13 : ssaka.
word 14 : kasmepiar.
word 15 : kenee.
word 16 : alox.
word 17 : maraylenna.
word 18 : ruga.
word 19 : byat.
```

Okay, I know. Some of these look... interesting. "khasleeshlangerranz"? Really? But hey, we also got "marie", "belo", "pen" - those sound legit!

Look, this is not that great, I know. Some names are way too long, some are just weird character combinations. But this is just the beginning. We're only using a window of 3 characters and a simple MLP. As we progress to better architectures (spoiler: transformers), we'll get WAY better results. Trust the process.

---

## Wrapping Up

So that's the Multi-Layer Perceptron! We went from looking at just 1 character (bigram) to looking at 3 characters (MLP with window size 3). Way better predictions, but still not perfect.

The limitation? We're still only looking at a **fixed window**. What if we could look at the **entire context** dynamically? That's where transformers come in, but that's a story for another blog.

For now, you've built a neural network that can generate names character by character. Not bad for a day's work!












