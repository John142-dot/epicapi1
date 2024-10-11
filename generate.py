import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# Load model (ensure this runs when the server starts)
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis1.6-Gemma2-9B",
    torch_dtype=torch.bfloat16,
    multimodal_max_length=8192,
    trust_remote_code=True
).cuda()

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

def generate_response(image_path, text):
    query = f'<image>\n{text}'
    
    # Preprocess inputs
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    
    # Generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output

def handler(request):
    if request.method == 'POST':
        data = request.json()
        image_path = data['image_path']  # Expecting image path from the request
        text = data['input']
        response_text = generate_response(image_path, text)
        return json.dumps({'output': response_text})
    else:
        return json.dumps({'error': 'Only POST requests are allowed.'})
