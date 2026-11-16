"""Native single-turn research supervision; no clipping or target repair."""
from personal_llm_data.research_chat import message_view

def encode_row(tokenizer,row,key,max_length=1536,max_new=64):
    messages=message_view(row['messages'],key)
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prompt)]!=prompt or len(full)<=len(prompt):raise ValueError('Research prompt/target boundary differs')
    if tokenizer.decode(full[len(prompt):],skip_special_tokens=True).strip()!=row['messages'][-1]['content'].strip():raise ValueError('Original supervision differs')
    if len(full)>max_length or len(full)-len(prompt)>max_new:raise ValueError('No research input truncation')
    return dict(input_ids=full,attention_mask=[1]*len(full),labels=[-100]*len(prompt)+full[len(prompt):]),prompt

class ChatSFTDataset:
    def __init__(self,records,tokenizer,max_length,model_key,max_new_tokens=64):
        self.rows=[encode_row(tokenizer,row,model_key,max_length,max_new_tokens)[0] for row in records]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.rows[index]
