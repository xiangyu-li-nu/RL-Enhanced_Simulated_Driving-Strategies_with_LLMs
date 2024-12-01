import json
from openai import OpenAI
from prompt_generate import *
from tqdm import tqdm
def generate_and_append_prompts(num_generations=1):
    # Creating an OpenAI Client
    client = OpenAI(
        base_url='',
        api_key=''  # Replace with your actual API key
    )

    # Used to store questions and answers
    qa_pairs = []

    for generation in tqdm(range(num_generations)):
        # Call the generated prompt function
        messages = generate_prompts_for_mav_rav(
            m=5, t=1.0, max_initial_speed=35, max_initial_accel=2, 
            allow_negative_distance=True, same_initial_conditions=False
        )

        # Iterate over the generated messages and get the model's answers
        for message in messages:
            try:
                # Calling OpenAI model
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": message[0]['content']},
                        {"role": "user", "content": message[1]['content']}
                    ]
                )

                # Get the model's answer
                response = completion.choices[0].message.content

                # Add questions and answers to a list
                qa_pairs.append({
                    "question": message[0]['content'] + "\n" + message[1]['content'],
                    "answer": response
                })

            except Exception as e:
                print(f"Error occurred while processing message: {message['content']}. Error: {e}")

                # If an error occurs, save the generated content to a file
                try:
                    with open('train.json', 'r') as f:
                        existing_data = json.load(f)
                except FileNotFoundError:
                    # If the file does not exist, initialize it to an empty list
                    existing_data = []

                # Append new questions and answers to existing data
                existing_data.extend(qa_pairs)

                # Write back to the file in append mode
                with open('train.json', 'w') as f:
                    json.dump(existing_data, f, indent=4)

                print(f"Results have been saved up to the current point in train.json.")
                return  # Jump out of the loop, end the program, and avoid continuing processing

    # If there are no errors, finally save all generated data
    try:
        with open('train.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If the file does not exist, initialize it to an empty list
        existing_data = []

    # Append new questions and answers to existing data
    existing_data.extend(qa_pairs)

    # Write back to the file in append mode
    with open('train.json', 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for {num_generations} generations appended to train.json")


# Call the function and specify the number of times to generate (for example, generate 5 times)
generate_and_append_prompts(num_generations=10)
