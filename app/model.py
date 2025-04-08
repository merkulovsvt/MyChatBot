import os
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification

from dotenv import load_dotenv
from ollama import chat, pull, generate

from app.rag import get_rag_response

load_dotenv()

'''Обычный инференс Phi-4-mini-instruct'''
# huggingface_hub.login(os.getenv('HUGGINGFACE_TOKEN'))
# model_name = "microsoft/Phi-4-mini-instruct"
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token_id = tokenizer.eos_token_id
#
# pipe = pipeline("text-generation",
#                 model=model,
#                 tokenizer=tokenizer)
#
#
# def get_response_from_messages_hf(messages, temperature=0.1):
#     generation_args = {
#         "max_new_tokens": 2000,
#         "return_full_text": False,
#         "temperature": temperature,
#     }
#
#     output = pipe(messages, **generation_args)
#     return output[0]['generated_text']

'''Ollama c Phi-4-mini-instruct (квантизация 8 бит)'''
# model_name = 'hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0'
# pull(model=model_name)
#
#
# def get_response_from_messages_hf(messages, temperature=0.7):
#     output = chat(
#         model=model_name,
#         messages=messages,
#         stream=False
#     )
#
#     return output['message']['content']

'''Ollama c Phi-4 (квантизация 5 бит)'''
# model_name = 'hf.co/bartowski/phi-4-GGUF:Q5_K_S'
# pull(model=model_name)
#
#
# def get_response_from_messages_hf(messages, temperature=0.7):
#     output = chat(
#         model=model_name,
#         messages=messages,
#         stream=False
#     )
#
#     return output['message']['content']


'''Используя TGI для LLama-3.2-3B-Instruct'''
# client = InferenceClient(model="http://127.0.0.1:8080")
#
#
# def get_response_from_messages_hf(messages, temperature=0.7):
#     output = client.chat_completion(
#         messages=messages,
#         max_tokens=1000,
#         temperature=temperature,
#     )
#
#     return output["choices"][0]["message"]["content"]


'''Phi-4-mini-instruct + rag'''
model_name = 'hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0'
# model_name = 'hf.co/bartowski/phi-4-GGUF:Q5_K_S'
pull(model=model_name)

rag_detection_prompt = """
<instructions>
You are a strict binary classifier for literary queries.

1. Analyze the user query inside <user_query> tags.
2. Respond with TRUE ONLY IF:
   - Query asks about book contents/plot/summary
   - Query asks about authors/biographical info
   - Query requests book recommendations
   - Query asks about literary analysis/themes
   - Query mentions specific books/authors/genres
3. Respond with FALSE for:
   - General conversations
   - Writing advice
   - Non-literary topics
   - Personal opinions
4. Respond ONLY with TRUE or FALSE in uppercase.
5. No explanations, punctuation or extra text.
</instructions>

<user_query>
{}
</user_query>
"""


def get_response_from_messages_hf(messages):
    rag_decision = generate(
        model=model_name,
        prompt=rag_detection_prompt.format(messages[-1]['content']),
    )['response'].strip()

    print(rag_decision)

    if rag_decision == 'TRUE':
        output = get_rag_response(messages)
    else:
        output = chat(
            model=model_name,
            messages=messages,
            stream=False
        )['message']['content']

    return output
