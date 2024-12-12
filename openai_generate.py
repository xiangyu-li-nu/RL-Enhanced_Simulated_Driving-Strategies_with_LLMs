import json
from openai import OpenAI
from prompt_generate import *
from tqdm import tqdm

def generate_and_append_prompts(num_generations=1):
    # Create OpenAI client
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key=''  # Replace with your actual API key
    )

    # List to store questions and answers
    qa_pairs = []

    for generation in tqdm(range(num_generations)):
        # Call the function to generate prompts
        messages = generate_prompts_for_mav_rav(
            m=5, t=1.0, max_initial_speed=35, max_initial_accel=2, 
            allow_negative_distance=True, same_initial_conditions=False
        )

        # Iterate through the generated messages and get model responses
        for message in messages:
            try:
                # Call OpenAI model
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": message[0]['content']},
                        {"role": "user", "content": message[1]['content']}
                    ]
                )

                # Get model response
                response = completion.choices[0].message.content

                # Append question and answer to the list
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
                    # If file does not exist, initialize as an empty list
                    existing_data = []

                # Append the newly generated questions and answers to existing data
                existing_data.extend(qa_pairs)

                # Save the updated data back to the file
                with open('train.json', 'w') as f:
                    json.dump(existing_data, f, indent=4)

                print(f"Results have been saved up to the current point in train.json.")
                return  # Exit the loop and end the program to avoid further processing

    # If no errors, finally save all the generated data
    try:
        with open('train.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # If file does not exist, initialize as an empty list
        existing_data = []

    # Append the newly generated questions and answers to existing data
    existing_data.extend(qa_pairs)

    # Save the updated data back to the file
    with open('train.json', 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results for {num_generations} generations appended to train.json")


# Call the function with a specified number of generations (e.g., generate 5 times)
generate_and_append_prompts(num_generations=2000)
