import json
from new_prompt_generate import * 

# Read JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Construct messages format
def prepare_messages(data):
    system_message = "You are an expert MAV decision-making assistant. Use the following input to predict MAV's action."

    messages_list = []
    for item in data:
        current_state_message = f"""
        MAV current state:
        - Position: {item['mav_state']['x_m_t']}, Lane: {item['mav_state']['lane_m_t']}, Speed: {item['mav_state']['v_m_t']}, Acceleration: {item['mav_state']['a_m_t']}

        RAV current state:
        - Position: {item['question'].split('Position: ')[1].split(',')[0]}, Lane: {item['question'].split('Lane: ')[1].split(',')[0]}, Speed: {item['question'].split('Speed: ')[1].split(',')[0]}, Acceleration: {item['question'].split('Acceleration: ')[1].split(',')[0]}

        Historical MAV behavior: {item['mav_history']}
        Reward detail: {item['reward_detail']}
        Available actions:
        - Decisions: {item['action_options']['MAV']['decisions']}
        - Cooperative Actions: {item['action_options']['MAV']['actions']['cooperate']}
        - Competitive Actions: {item['action_options']['MAV']['actions']['compete']}
        Question:
        {item['question']}
        """

        messages_list.append([
            {"role": "system", "content": system_message},
            {"role": "user", "content": current_state_message}
        ])

    return messages_list

# Call GPT model to generate results
def generate_content_openai(prompt):
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-Gdfbmg3qffLf5ZH0RHYNAjpgdExFUC0uKoY8oh2K7q8oIBJI'  # Replace with your actual API key
    )

    # Generate MAV decision
    mav_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": ""},
                  {"role": "user", "content": prompt}]
    )
    decision = mav_completion.choices[0].message.content
    decision_action = extract_wrapped_content(decision)

    return decision, decision_action

def generate_content_llama(prompt):
    peft_model_id = "peft_model"  # Replace with the path to your trained model
    config = PeftConfig.from_pretrained(peft_model_id)

    # Load base model (e.g., T5, BART, or other Seq2Seq models)
    model_name = "D:/预训练模型/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load PEFT model (i.e., trained LoRA model)
    model = PeftModel.from_pretrained(model, peft_model_id)

    # Load tokenizer (ensure compatibility with the model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Generate MAV decision
    outputs = pipe([
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ], max_new_tokens=4096)

    decision = outputs[0]["generated_text"][2]["content"]
    decision_action = extract_wrapped_content(decision)

    return decision, decision_action

def process_data(data):
    t = 1
    all_reward = []
    chioce = "llama"

    for item in data:
        # Get MAV current state
        mav_state = item['mav_state']
        mav_x = mav_state['x_m_t']
        mav_lane = mav_state['lane_m_t']
        mav_v = mav_state['v_m_t']
        mav_a = mav_state['a_m_t']

        # Get RAV current state
        rav_state = item['rav_state']
        rav_x = rav_state['x_r_t']
        rav_lane = rav_state['lane_r_t']
        rav_v = rav_state['v_r_t']
        rav_a = rav_state['a_r_t']

        # Get action options
        question = item["question"]

        # Extract question text
        if chioce == "openai":
            decision, decision_action = generate_content_openai(question)

        if chioce == "llama":
            decision, decision_action = generate_content_llama(question)

        # May be MAV or RAV
        any_decision = decision_action["decision"].strip().lower()
        any_action = decision_action["action"].strip().lower()

        if any_action == None:
            any_action = "accelerate"
            print("No action, adding default action accelerate")

        # Calculate time delay before and after initial acceleration
        time_before_mav = calculate_time_to_zero(mav_x, mav_v, mav_a)
        time_before_rav = calculate_time_to_zero(rav_x, rav_v, rav_a)

        if item["type"] == "mav":
            # Update acceleration
            if any_action == 'accelerate':
                mav_a += 2 * t  # Increase acceleration
            elif any_action == 'decelerate':
                mav_a -= 2 * t  # Decrease acceleration
        else:
            if any_action == 'accelerate':
                rav_a += 2 * t  # Increase acceleration
            elif any_action == 'decelerate':
                rav_a -= 2 * t  # Decrease acceleration