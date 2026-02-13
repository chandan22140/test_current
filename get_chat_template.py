"""
Standalone chat template utility for applying chat templates to tokenizers.
No external dependencies beyond transformers.
Supports: gemma3, chatml, llama3, and custom templates.
"""

# Gemma 3 chat template (from unsloth)
GEMMA3_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\n' }}"
    "{% endif %}"
)

# ChatML template (used by many models)
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# Llama 3 template
LLAMA3_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

# Generic fallback template
GENERIC_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

CHAT_TEMPLATES = {
    "gemma3": GEMMA3_TEMPLATE,
    "gemma": GEMMA3_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
    "llama-3": LLAMA3_TEMPLATE,
    "generic": GENERIC_TEMPLATE,
}


def get_chat_template(tokenizer, chat_template="gemma3"):
    """
    Apply a chat template to a tokenizer.
    
    Args:
        tokenizer: A HuggingFace tokenizer instance.
        chat_template: Name of the template ("gemma3", "chatml", "llama3", "generic")
                       or a raw Jinja2 template string.
    
    Returns:
        The tokenizer with the chat_template attribute set.
    """
    if chat_template in CHAT_TEMPLATES:
        template_str = CHAT_TEMPLATES[chat_template]
    else:
        # Assume it's a raw Jinja2 template string
        template_str = chat_template

    tokenizer.chat_template = template_str

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
