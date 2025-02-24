from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from tqdm import tqdm
import re
import os
import pickle


def load_data():

    path = "data/gsm/processed_gsm.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    print(data[:5])

    return data


def load_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    hf_token: str = "",
    ):

    if hf_token != "":
        login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return model, tokenizer, device


def format_prompt(question: str) -> str:
        """
        Format the question into a prompt for Mistral-7B-Instruct.
        """
        return f"""<s>[INST] Solve this math problem step by step. Show your work clearly and end your solution with '#### X' where X is your final numerical answer with no units: {question} [/INST]"""


@torch.no_grad()
def get_model_response(model, tokenizer, device, prompts: List[str]) -> List[str]:
    """
    Get responses from the local Mistral model for a batch of prompts.
    """
    inputs = tokenizer(prompts, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Remove the prompt from each response
    responses = [response[len(prompt):] for response, prompt in zip(responses, prompts)]
    
    return responses


def extract_final_answer(response: str) -> float:
    """
    Extract the final numerical answer from the model's response.
    """
    patterns = [
        r"#### (\-?\d*\.?\d+)",
        r"The answer is (\-?\d*\.?\d+)",
        r"= (\-?\d*\.?\d+)(?!\d)",
        r"(\-?\d*\.?\d+)(?:\s*|$)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    
    raise ValueError("No numerical answer found in response")


def evaluate_response(correct_answer: float, model_response: str) -> Tuple[bool, float]:
    """
    Compare the model's response to the correct answer.
    """
    try:
        predicted_answer = extract_final_answer(model_response)
        is_correct = abs(predicted_answer - correct_answer) < 1e-6
        return is_correct, predicted_answer
    except ValueError:
        return False, None


def evaluate_dataset(model, tokenizer, device, dataset, batch_size) -> Dict:

    results = {
        'correct': 0,
        'total': 0,
        'accuracy': 0.0,
        'wrong_predictions': [],
        'correct_predictions': []
    }

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        prompts = [format_prompt(item['question']) for item in batch]

        try:
            responses = get_model_response(model, tokenizer, device, prompts)
            
            for item, response in zip(batch, responses):
                is_correct, predicted = evaluate_response(
                    float(item['answer']), 
                    response
                )
                
                results['total'] += 1
                if is_correct:
                    results['correct'] += 1
                    results['correct_predictions'].append({
                        'question': item['question'],
                        'correct_answer': item['answer'],
                        'predicted_answer': predicted,
                        'full_response': response
                    })
                else:
                    results['wrong_predictions'].append({
                        'question': item['question'],
                        'correct_answer': item['answer'],
                        'predicted_answer': predicted,
                        'full_response': response
                    })
                
                # Print progress more frequently
                if results['total'] % 10 == 0:
                    current_accuracy = results['correct'] / results['total']
                    print(f"\nCurrent accuracy: {current_accuracy:.2%} ({results['correct']}/{results['total']})")
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

        # Only calculate accuracy if we have processed at least one example
        if results['total'] > 0:
            results['accuracy'] = results['correct'] / results['total']
        else:
            print("Warning: No examples were successfully processed")
            results['accuracy'] = 0.0

    return results


def main():

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the HuggingFace token from the environment variables
    hf_token = os.getenv("HF_TOKEN")

    dataset = load_data()
    model, tokenizer, device = load_model(hf_token=str(hf_token))
    results = evaluate_dataset(model, tokenizer, device, dataset, batch_size=16)

if __name__ == "__main__":

    main()