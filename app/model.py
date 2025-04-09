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


'''Phi-4-mini-instruct + RAG'''
model_name = 'hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q8_0'
pull(model=model_name)

rag_detection_prompt = """
<instruction>
Ты — строгий бинарный классификатор литературных запросов. Твоя задача — определить, относится ли запрос к литературной тематике.

1. Проанализируй текст внутри тегов <user_query>.
2. Ответь ТОЛЬКО "TRUE" (заглавными буквами), если запрос относится К ХОТЯ БЫ ОДНОЙ из следующих категорий:
   - Обсуждение содержания, сюжета или краткого пересказа книг
   - Вопросы о писателях, их жизни, творчестве или биографии
   - Просьбы порекомендовать книги, авторов или произведения
   - Литературный анализ: темы, идеи, символы, художественные приёмы
   - Упоминание конкретных книг, авторов, жанров или литературных направлений
3. Ответь "FALSE", если запрос:
   - Общий, абстрактный или не имеет отношения к литературе
   - О написании текстов, сочинений, эссе, сценариев и т.п.
   - Относится к другим видам искусства, поп-культуре, философии, психологии и т.д.
   - Является личным мнением, размышлением, шуткой или разговором "ни о чём"
4. Отвечай СТРОГО одним словом: TRUE или FALSE (заглавными буквами).
5. Не добавляй никаких объяснений, знаков препинания или комментариев.
</instruction>

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

    if 'TRUE' in rag_decision:
        output = get_rag_response(messages)
    else:
        output = chat(
            model=model_name,
            messages=messages,
            stream=False
        )['message']['content']

    return output
