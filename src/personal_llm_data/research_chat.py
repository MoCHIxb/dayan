"""Explicit versioned message views for research, without editing source records."""
POLICIES={
    'qwen3_8b':'native_qwen_no_thinking_v1',
    'mistral7b':'mistral_system_prefix_in_user_v1',
    'phi4mini':'native_phi_v1',
}

def message_view(messages,model_key=None):
    if model_key is not None and model_key not in POLICIES:raise ValueError('Unregistered chat policy')
    copied=[dict(message) for message in messages]
    if model_key!='mistral7b':return copied
    # Current research records are one system instruction, one real request,
    # and optionally a real target. Reject other formats rather than guessing.
    roles=[m.get('role') for m in copied]
    if roles not in (['system','user'],['system','user','assistant']):
        raise ValueError('Mistral policy is fixed for single-turn system/user[/assistant] records')
    if any(not isinstance(m.get('content'),str) for m in copied):raise ValueError('Text messages required')
    instruction=copied[0]['content'];request=copied[1]
    request['content']=instruction+'\n\n'+request['content']
    return copied[1:]
