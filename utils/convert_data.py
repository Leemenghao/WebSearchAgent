import json
import os

def process_file(file_path):
    temp_file_path = file_path + '.tmp'
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(temp_file_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Perform the transformation
                # 1. Rename 'id' to 'chat_history' (or add chat_history and remove id)
                # 2. Set type to array
                
                if 'id' in data:
                    # We remove 'id' and add 'chat_history' initialized as an empty list
                    # preserving the order if possible, though dicts in Python 3.7+ represent insertion order
                    
                    new_data = {}
                    # If we want 'chat_history' to be first where 'id' was
                    for key, value in data.items():
                        if key == 'id':
                            new_data['chat_history'] = []
                        else:
                            new_data[key] = value
                    
                    data = new_data
                    
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON lines: {e}")
                continue

    # Replace original file with processed file
    os.replace(temp_file_path, file_path)
    print(f"Successfully processed {file_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'question.jsonl')
    
    if os.path.exists(data_file):
        process_file(data_file)
    else:
        print(f"File not found: {data_file}")
