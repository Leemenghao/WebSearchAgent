import json
import os

def convert_answer_file(input_path, output_path):
    print(f"Reading from {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} items")
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in data:
                # Extract id from line_number and answer from answer
                # If line_number is missing, we might need a fallback, but assuming it exists based on context
                record = {
                    "id": item.get("line_number"),
                    "answer": item.get("answer")
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        print(f"Successfully wrote to {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    base_dir = "/home/lmh/lee/competition/ali_agent_2602/submit"
    input_file = os.path.join(base_dir, "answer.json")
    output_file = os.path.join(base_dir, "answer.jsonl")
    
    if os.path.exists(input_file):
        convert_answer_file(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
