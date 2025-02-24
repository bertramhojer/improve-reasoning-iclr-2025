from datasets import load_dataset
import re
import pickle


def load_gsm(split: str = "test"):
    """
    Load the GSM8K dataset using the Hugging Face datasets library.
    """
    dataset = load_dataset("openai/gsm8k", "main")[split]
    processed_data = []
    
    for item in dataset:
        try:
            answer = extract_answer_from_solution(item["answer"])
            processed_data.append({
                "question": item["question"],
                "answer": answer,
                "full_solution": item["answer"]
            })
        except ValueError as e:
            print(f"Skipping question due to parsing error: {item['question'][:100]}...")
            continue
    
    if not processed_data:
        raise ValueError("No valid questions found in dataset")
    
    print(f"Successfully processed {len(processed_data)} questions")
    return processed_data


def extract_answer_from_solution(solution: str) -> float:
    """
    Extract the final answer from GSM8K solution string.
    
    Args:
        solution: Solution string from GSM8K dataset
        
    Returns:
        Final answer as float
    """
    patterns = [
        r"#### (\-?\d*\.?\d+)",
        r"The answer is (\-?\d*\.?\d+)",
        r"= (\-?\d*\.?\d+)(?!\d)",
        r"(\-?\d*\.?\d+)(?:\s*|$)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, solution)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    
    print(f"Could not extract answer from solution: {solution}")
    raise ValueError("Could not extract answer from solution")


if __name__ == "__main__":
    data = load_gsm()
    output_file = "data/gsm8k/processed_gsm.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Data saved to {output_file}")