# Debug Script - Check Claude API Key in Container

# Test Claude API key directly in container
docker exec $(docker ps -q --filter ancestor=rag-app:local) python -c "
import os
import anthropic

api_key = os.getenv('ANTHROPIC_API_KEY')
print(f'API Key Found: {bool(api_key)}')
print(f'API Key Length: {len(api_key) if api_key else 0}')
print(f'API Key Starts With: {api_key[:10] if api_key else \"None\"}...')

if api_key:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=10,
            messages=[{'role': 'user', 'content': 'test'}]
        )
        print('✅ Claude API Works!')
        print(f'Response: {response.content[0].text}')
    except Exception as e:
        print(f'❌ Claude API Error: {type(e).__name__}: {e}')
else:
    print('❌ API Key not found!')
"

