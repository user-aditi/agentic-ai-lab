# Practical Experiment — Lab 1: Fine-Tuning an Image Captioning Model

**Subject:** Agentic AI  
**Students:** Harshit Arora & Aditi Jha

---

## A. Problem Statement

Pre-trained vision-language models such as BLIP (Bootstrapping Language-Image Pre-training) provide general-purpose image captioning capabilities, but their outputs may not be accurate or domain-specific enough for specialized use cases. For instance, a generic captioning model may not correctly describe football-related images with the appropriate terminology and context.

The objective of this practical is to **fine-tune a pre-trained BLIP image captioning model** (`Salesforce/blip-image-captioning-base`) on a domain-specific football image dataset so that the model learns to generate accurate, contextually relevant captions for football images. The experiment demonstrates the complete fine-tuning pipeline — from loading a dataset and preparing a PyTorch DataLoader, to training the model and running inference to verify improved caption quality.

---

## B. Solution

### Step 1: Set Up the Environment

Install the required libraries — HuggingFace `transformers` (from the latest GitHub main branch) and `datasets` — on Google Colab with GPU runtime enabled.

### Step 2: Load the Image Captioning Dataset

Load the `ybelkada/football-dataset` from the HuggingFace Hub using the `datasets` library. This is a small, curated dataset containing football-related images paired with descriptive text captions. Verify the dataset by inspecting the caption and image of the first example.

### Step 3: Create a PyTorch Dataset Class

Define a custom `ImageCaptioningDataset` class that wraps the HuggingFace dataset. In the `__getitem__` method, each image-text pair is processed using the BLIP processor which:
- Converts the image into pixel values.
- Tokenizes the caption text.
- Applies padding to `max_length` and returns PyTorch tensors.

The batch dimension is squeezed before returning.

### Step 4: Load the Pre-trained Model and Processor

Load the pre-trained BLIP model (`Salesforce/blip-image-captioning-base`) and its corresponding `AutoProcessor` from HuggingFace. The processor handles both image preprocessing (resizing, normalization) and text tokenization in a single step.

### Step 5: Prepare the DataLoader

Instantiate the custom `ImageCaptioningDataset` with the loaded dataset and processor, and wrap it in a PyTorch `DataLoader` with:
- **Batch size:** 2
- **Shuffle:** True (for randomized training order)

### Step 6: Train (Fine-Tune) the Model

Fine-tune the BLIP model for **50 epochs** using the following configuration:
- **Optimizer:** AdamW with learning rate `5e-5`
- **Device:** CUDA GPU (if available, else CPU)
- **Loss:** The model's built-in cross-entropy loss computed by passing `input_ids` as both input and labels

In each epoch, iterate over all batches, perform a forward pass, compute the loss, backpropagate gradients, and update model weights. The training loss is printed at each step to monitor convergence.

### Step 7: Run Inference on a Single Image

After training, test the fine-tuned model on a sample image from the dataset:
1. Load and display the image.
2. Process it using the BLIP processor to obtain pixel values.
3. Use `model.generate()` with `max_length=50` to produce a caption.
4. Decode the generated token IDs into a human-readable caption string using `processor.batch_decode()`.

### Step 8: Load and Evaluate a Pre-Finetuned Model from the Hub

Load the already fine-tuned model (`ybelkada/blip-image-captioning-base-football-finetuned`) from the HuggingFace Hub to compare or verify results.

### Step 9: Visualize Results on the Entire Dataset

Generate captions for all images in the dataset using the fine-tuned model and display the results in a **2×3 matplotlib grid**:
- Each subplot shows a football image.
- The title of each subplot displays the model's generated caption.
- Axes are turned off for a clean visual presentation.

---

## C. Result

> **Screenshot 1 (Training Loss Output):**
> A screenshot of the notebook output showing the training loop across 50 epochs. Each epoch prints the epoch number and the loss value for each batch. The loss values should show a decreasing trend — starting from a higher value (e.g., ~1.5–2.0) in early epochs and converging to a lower value (e.g., ~0.2–0.4) in later epochs, indicating successful fine-tuning.

> **Screenshot 2 (Single Image Inference):**
> A screenshot showing the inference output for a single football image. The notebook displays the football image followed by the generated caption text printed below it (e.g., *"a football player is kicking the ball"*), demonstrating that the fine-tuned model produces relevant, domain-specific captions.

> **Screenshot 3 (Grid Visualization of All Captions):**
> A screenshot of the 2×3 matplotlib figure displaying all images from the football dataset. Each image is shown in its own subplot with the generated caption as the title (e.g., *"Generated caption: two soccer players are playing a game"*). The captions are contextually accurate and reflect football-specific terminology, confirming the model has been successfully fine-tuned on the domain-specific dataset.
