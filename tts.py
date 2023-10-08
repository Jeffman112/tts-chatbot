import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, StoppingCriteria, StoppingCriteriaList, AutoProcessor, BarkModel
from torch import cuda
import soundfile as sf
import pygame
from txtai.pipeline import TextToSpeech
import re

pygame.init()
tts = TextToSpeech()
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name()
print(f"Using device: {device} ({device_name})")
MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

streamer = TextStreamer(tokenizer)
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

class StopOnEOSToken(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        eos_token_id = tokenizer.eos_token_id
        return eos_token_id in input_ids[0]

stopping_criteria = StoppingCriteriaList([StopOnEOSToken()])
while True:
    prompt = input("Prompt: ")
    print("Generating...")
    prompt = prompt.replace("?", "")
    prompt = prompt.replace(",", "")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace("!", "")
    formatted_prompt = f"<s>[INST]Make sure your response is short and simple: {prompt}[/INST]"
    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        streamer=streamer
    )


    for seq in sequences:
        result = seq['generated_text']
        result = result.replace(formatted_prompt, "")
        result = re.sub(r'\*[^*]*\*', '', result)
        result = result.replace("...", ".")
        result = result.strip()
        print(f"Generating Speech....")
    speech = tts(result)
    sf.write("speech.wav", speech, 22050)
    print("Speaking...")
    sound = pygame.mixer.Sound("speech.wav")
    sound.play()
    pygame.mixer.music.stop()
