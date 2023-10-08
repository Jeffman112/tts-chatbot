# Text-to-Speech LLM
### This is a Python script that uses the Hugging Face Transformers library to create a text-to-speech chatbot. The chatbot takes user input, generates text responses using a pre-trained language model, and converts these responses into speech that can be played back using the Pygame library.

## Prerequisites
Before running this script, make sure you have the following libraries and resources installed:

PyTorch (GPU support recommended)
Hugging Face Transformers
Soundfile
Pygame
txtai (for text-to-speech conversion)
Pre-trained language model weights and tokenizer (specified in MODEL_NAME)

Run the `tts.py` script in your Python environment.

The chatbot will prompt you for input. Enter a text prompt, and the chatbot will generate a response and play it back as speech.
## Configuration
You can modify the top_k, top_p, num_return_sequences, and other parameters in the pipeline function to adjust the text generation behavior.
The script includes a stopping criteria that stops text generation when the end-of-sentence token is encountered. You can customize this criteria or remove it as needed.
The chatbot also performs basic text formatting to remove unwanted characters and tokens from the generated response.
## Notes
The chatbot uses the txtai library for text-to-speech conversion. Ensure that you have it installed and configured properly for text-to-speech functionality.
The script assumes that you have a compatible GPU for faster model inference. If not, it will fall back to CPU inference.
Make sure to respect any usage terms and conditions associated with the pre-trained language model you choose to use.
## Disclaimer
This script uses a pre-trained language model and may generate text responses that do not always align with expectations. It is important to review and filter the generated content, especially in cases where the model's output may be inappropriate or undesirable.
