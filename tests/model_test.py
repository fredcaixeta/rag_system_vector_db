import transformers
import torch

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
pipe("Hey how are you doing today?")

