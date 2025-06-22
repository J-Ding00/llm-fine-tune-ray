# Only use when data is very limited
import yaml
import json
from api.clients import openai_client
import random
import os

# def generate_transcript(criteria, num_per_criteria_category, gpt_model):
#     """
#     Generates a synthetic speech transcript using the given GPT model based on a provided prompt.
#     """

#     good_prompt = f"""
#     Write me a paragraph with some {criteria}, only output the paragraph itself.
#     """

#     bad_prompt = f"""
#     Write me a paragraph that lacks of {criteria}, only output the paragraph itself.
#     """
#     transcripts = []
#     for _ in range(num_per_criteria_category):
#         response = openai_client.responses.create(
#             model=gpt_model,
#             input=good_prompt,
#             temperature=temperature,
#             max_output_tokens=max_tokens
#         )
#         transcripts.append(response.output_text.strip())

#         response = openai_client.responses.create(
#             model=gpt_model,
#             input=bad_prompt,
#             temperature=temperature,
#             max_output_tokens=max_tokens
#         )
#         transcripts.append(response.output_text.strip())
#     return transcripts

def generate_negative_formality_transcript_samples(n, model, real_samples, output_folder):
    """
    Generate `n` long-form samples recursively from GPT-4o-mini with chat history.
    """

    system_prompt = "You are an assistant with a persona who rewrites reference sample for training a communication evaluation model. Do not include any introductory phrases or openings; start directly with the main content."
    output_file = os.path.join(output_folder, 'generated_bad_formality.jsonl')

    with open(output_file, "a") as f:
        for i in range(1, n + 1):
            
            try:
                sample = random.choice(real_samples)

                user_prompt = f"""
                You are given a reference article-style sample.

                Imitate a casual blogger who uses informal language, contractions, or slang inappropriately in a serious context, rewrite the sample in a way that **lacks formality**.

                Only output the transcript content.

                ### Reference Sample:
                {sample.strip()}
                """.strip()

                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, 
                                       {"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_tokens=1024
                )

                assistant_reply = response.choices[0].message.content.strip()
                f.write(json.dumps({'id':i, 'trait':'formality', 'text':assistant_reply, 'original':sample}) + "\n")


            except Exception as e:
                print(f"Error at sample {i}: {e}")
                break

def generate_negative_persuasion_transcript_samples(n, model, real_samples, output_folder):
    """
    Generate `n` long-form samples recursively from GPT-4o-mini with chat history.
    """

    system_prompt = "You are an assistant with a persona who rewrites reference sample for training a communication evaluation model. Do not include any introductory phrases or openings; start directly with the main content."
    output_file = os.path.join(output_folder, 'generated_bad_persuasion.jsonl')

    with open(output_file, "a") as f:
        for i in range(1, n + 1):
            
            try:
                sample = random.choice(real_samples)

                user_prompt = f"""
                You are given a reference article-style sample.

                Imitate a passive commentator who avoids strong arguments and fails to support claims with logic or evidence, rewrite the sample in a way that **lacks persuasiveness**.

                Only output the transcript content.

                ### Reference Sample:
                {sample.strip()}
                """.strip()

                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, 
                                       {"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_tokens=1024
                )

                assistant_reply = response.choices[0].message.content.strip()
                f.write(json.dumps({'id':i, 'trait':'persuasiveness', 'text':assistant_reply, 'original':sample}) + "\n")


            except Exception as e:
                print(f"Error at sample {i}: {e}")
                break

def generate_negative_transition_logic_transcript_samples(n, model, real_samples, output_folder):
    """
    Generate `n` long-form samples recursively from GPT-4o-mini with chat history.
    """

    system_prompt = "You are an assistant with a persona who rewrites reference sample for training a communication evaluation model. Do not include any introductory phrases or openings; start directly with the main content."
    output_file = os.path.join(output_folder, 'generated_bad_transition_logic.jsonl')

    with open(output_file, "a") as f:
        for i in range(1, n + 1):
            
            try:
                sample = random.choice(real_samples)

                user_prompt = f"""
                You are given a reference article-style sample.
 
                Imitate a disorganized writer who jumps between ideas without smooth transitions or coherent flow, rewrite the sample in a way that **lacks clear transition logic**.

                Only output the transcript content.

                ### Reference Sample:
                {sample.strip()}
                """.strip()

                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, 
                                       {"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_tokens=1024
                )

                assistant_reply = response.choices[0].message.content.strip()
                f.write(json.dumps({'id':i, 'trait':'transition logic', 'text':assistant_reply, 'original':sample}) + "\n")


            except Exception as e:
                print(f"Error at sample {i}: {e}")
                break

def generate_negative_filler_words_transcript_samples(n, model, real_samples, output_folder):
    """
    Generate `n` long-form samples recursively from GPT-4o-mini with chat history.
    """

    system_prompt = "You are an assistant with a persona who rewrites reference sample for training a communication evaluation model. Do not include any introductory phrases or openings; start directly with the main content."
    output_file = os.path.join(output_folder, 'generated_bad_filler_words.jsonl')

    with open(output_file, "a") as f:
        for i in range(1, n + 1):
            
            try:
                sample = random.choice(real_samples)

                user_prompt = f"""
                You are given a reference article-style sample.

                Imitate a writer who uses redundant connectors, rewrite the sample in a way that **includes filler words rarely but surely**.

                Only output the transcript content.

                ### Reference Sample:
                {sample.strip()}
                """.strip()

                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, 
                                       {"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_tokens=1024
                )

                assistant_reply = response.choices[0].message.content.strip()
                f.write(json.dumps({'id':i, 'trait':'filler words', 'text':assistant_reply, 'original':sample}) + "\n")


            except Exception as e:
                print(f"Error at sample {i}: {e}")
                break

if __name__ == "__main__":
    config_path = 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tg_config = config['raw_transcripts_gen']
    num_samples = tg_config['num_samples']
    output_folder = tg_config['output_folder']
    gpt_model = config['openai']['generate_model']
    real_sample_path = config['load_data']['raw_path']

    with open(real_sample_path, "r") as f:
        real_samples = [json.loads(l)['text'].strip() for l in f]

    generate_negative_formality_transcript_samples(num_samples, gpt_model, real_samples, output_folder)
    generate_negative_transition_logic_transcript_samples(num_samples, gpt_model, real_samples, output_folder)
    generate_negative_persuasion_transcript_samples(num_samples, gpt_model, real_samples, output_folder)
    generate_negative_filler_words_transcript_samples(num_samples, gpt_model, real_samples, output_folder)

    