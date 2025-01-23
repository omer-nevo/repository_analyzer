import openai
import asyncio
import nest_asyncio  # for Interactive Environments (Pycharm,Vscode etc)
from src.utils.rate_limiter import get_rate_limiter
from src.utils.config import get_openai_key  # Load OpenAI API key from config

nest_asyncio.apply()
# Initialize OpenAI Async Client
client = openai.AsyncOpenAI(api_key=get_openai_key())


class OpenAIAssistant:
    """An OpenAI Assistant for code analysis and repository queries."""

    def __init__(self, assistant_id=None):
        self.assistant_id = assistant_id

    async def create_assistant(self):
        """Create an OpenAI Assistant (only needed once)."""
        async with get_rate_limiter():
            assistant = await client.beta.assistants.create(
                name="Code Analysis Assistant",
                instructions="You are a code analysis assistant. Help users understand and query repository code.",
                model="gpt-4-turbo",
                tools=[{"type": "code_interpreter"}]
            )
        self.assistant_id = assistant.id
        return self.assistant_id

    async def create_thread(self):
        """Create a new conversation thread."""
        async with get_rate_limiter():
            thread = await client.beta.threads.create()
        return thread.id

    async def ask_question(self, thread_id: str, question: str):
        """Send a query to the Assistant."""
        if not self.assistant_id:
            raise ValueError("Assistant has not been created yet!")

        async with get_rate_limiter():
            message = await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=f"Answer this question: {question}."
                        f" Provide clear, formatted code snippets in your responses if needed."
            )

        # Run the assistant to get a response
        return await self.run_assistant(thread_id)

    async def run_assistant(self, thread_id: str):
        """Run the assistant and fetch responses."""
        async with get_rate_limiter:
            run = await client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )

        # Wait for completion
        while True:
            async with get_rate_limiter:
                run_status = await client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )
            if run_status.status == "completed":
                break
            # TODO: elif handle more status codes
            await asyncio.sleep(1)  # Polling delay

        # Get messages
        async with get_rate_limiter:
            messages = await client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value
