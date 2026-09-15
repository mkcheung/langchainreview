# Lesson: Agents Under the Hood
#
# This file demonstrates the agent loop by replacing LangChain abstractions
# with raw ollama SDK calls, so you can see exactly what LangChain does for you.
#
# How to run:
#   uv run python3 1_agent_loop_langchain_tool_calling.py

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import ollama
from langsmith import traceable

MAX_INTERATIONS = 10
MODEL = "qwen3:1.7b"

# --- LESSON: Tool functions as plain Python ---
# The @tool decorator (commented out) would auto-generate JSON schemas from
# type hints and docstrings, and wrap these as LangChain StructuredTool objects.
# Here we use plain functions so we can see those two responsibilities separately:
# the function logic stays here, and the schema is written by hand below.

# @tool
@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the catalog """
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard":89.50}
    return prices.get(product, 0)

# @tool
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier:str) -> float:
    """ Apply a discont tier to a price and return the final price.
        Available tiers: bronze, silver, gold."""
    print(f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# --- LESSON: Manual JSON tool schemas ---
# This is what @tool generates automatically. Every LLM API (OpenAI, Anthropic,
# Ollama, etc.) requires tools to be described as JSON so the model knows what
# arguments to pass. LangChain reads your type hints and docstring and produces
# exactly this structure — we're just writing it by hand to see it clearly.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog",
            "parameters": {
                "product": {
                    "type":"string",
                    "description": "The product name, e.g. 'laptop', 'headphnes', 'keyboard'",
                },
            },
            "required": ["product"]
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# --- LESSON: Provider-specific LLM call ---
# ollama.chat() is a direct call to a locally running Ollama instance.
# LangChain's llm_with_tools.invoke() does the same thing but is provider-agnostic —
# swap init_chat_model("ollama:...") for init_chat_model("openai:...") and the
# rest of your code stays identical. Here we use ollama directly to see what's
# actually going over the wire.
@traceable(name="Ollama Chat", run_type='llm')
def ollama_chat_trace(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    # tools_dict = (t.name: t for t in tools)
    # llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm_with_tools = llm.bind_tools(tools)

    # --- LESSON: tools_dict is the dispatcher ---
    # The LLM can only return a tool name as a string. This dict maps that string
    # back to the actual Python function to call. LangChain builds this internally;
    # here we build it ourselves so the mechanism is visible.
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # --- LESSON: Raw message dicts instead of LangChain message objects ---
    # LangChain's HumanMessage/SystemMessage/ToolMessage are just thin wrappers
    # around the {"role": "...", "content": "..."} format that all LLM APIs use
    # natively. Using plain dicts here exposes that underlying protocol.
    messages = [
        # SystemMessage(
        #     content=(
        #         "You are a helpful shopping assistant. "
        #         "You have access to a product catalog tool "
        #         "and a discount tool.\n\n"
        #         "STRICT RULES = ypu must follow these exactly:\n"
        #         "1. NEVER guess or assume any product price. "
        #         "You MUST call get_product_price first to get the real price"
        #         "2. Only call apply_discount AFTER you have received "
        #         "a price from get_product_price. Pass the exact price "
        #         "returned by get_product_price - do NOT pass a made-up number.\n"
        #         "3. NEVER calculate discounts yourself using main. "
        #         "Always use the apply_discount tool.\n"
        #         "4. If the user does not specify a discount tier, "
        #         "ask them which tier to use - do NOT assume one."
        #     )
        # ),
        {
            "role":"system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES = ypu must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using main. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            ),
        },
        # HumanMessage(content=question)
        {"role":"user", "content": question},
    ]

    # --- LESSON: This loop IS the agent ---
    # There is no magic inside LangChain agents — it's this: call the LLM, check
    # if it wants to use a tool, run the tool, append the result to the message
    # history, and repeat until the model returns a plain text answer. The message
    # history is how the model "remembers" what tools it already called and what
    # they returned.
    for iteration in range(1, MAX_INTERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        # ai_message = llm_with_tools.invoke(messages)

        # tool_calls = ai_message.tool_calls

        response = ollama_chat_trace(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        tool_call = tool_calls[0]

        # --- LESSON: Raw tool call response format ---
        # LangChain normalizes tool call responses into a flat dict with "name",
        # "args", and "id" keys regardless of provider. Ollama's native format uses
        # an object: tool_call.function.name / tool_call.function.arguments.
        # Both carry the same data — LangChain just standardizes the shape.
        # tool_name = tool_call.get("name")
        # tool_args = tool_call.get("args", {})
        # tool_call_id = tool_call.get("id")
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f".  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # --- LESSON: Calling the tool directly vs via LangChain ---
        # When tools are plain functions, we call them with **tool_args directly.
        # If they were @tool StructuredTool objects, we'd use tool_to_use.invoke(tool_args)
        # instead, which adds LangChain's input validation and error handling on top.
        # observation = tool_to_use.invoke(tool_args)
        observation = tool_to_use(**tool_args)

        print(f"    [Tool Result] {observation}")
        messages.append(ai_message)

        # --- LESSON: Feeding the result back as a "tool" role message ---
        # The model needs to see what the tool returned so it can decide what to do
        # next. LangChain's ToolMessage is a wrapper around this same dict with
        # role="tool". Appending it to messages is how the agent loop "remembers"
        # the tool result on the next iteration.
        # messages.append(
        #     ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        # )
        messages.append(
            {
                "role":"tool",
                "content":str(observation),
            }
        )
    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent('What is the price of a laptop after applying a gold discount?')
