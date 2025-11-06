import os
import cohere
import json


api_key = os.environ.get("COHERE_API_KEY")
if not api_key:
    raise Exception("COHERE_API_KEY environment variable not set.")

def get_drink_details() -> dict[str,dict[str,dict[str,list]]]:
    """Get and return all drinks and their details from the drink_descriptions.json file.

    Raises:
        Exception: If there was an error while loading from file.

    Returns:
        dict[str,dict[str,dict[str,list]]]: Drink descriptions from JSON, in the format:

                                            {
                                                "<category>": {
                                                    "<drink name>": {
                                                        "taste": [<list of taste notes>]
                                                    }
                                                }
                                            }
    """
    try:
        with open("drink_descriptions.json", "r") as f:
            descriptions = json.load(f)
    except (FileNotFoundError, EOFError):
        raise Exception("Drink descriptions file not found or is empty.")

    return descriptions

def get_descriptions_formatted(drink_details) -> str:
    """
    Format drink descriptions into a string so model can see the menu.
    Returns a string in the format of:
    <category>
      <drink name>: <taste notes>
      <drink name>: <taste notes>
      ...
    ...
    """
    fin = ""
    for category, category_drinks in drink_details.items():
        fin += f"{category.capitalize()} Drinks:\n"
        for drink_name, drink_info in category_drinks.items():
            taste_notes = ", ".join(drink_info.get("taste", []))
            fin += f" - {drink_name}: {taste_notes}\n"
    
    return fin


def order_drink():
    """Start a conversation with the chatbot to order a drink.
    """
    drink_list = get_drink_details()
    sysprompt_ordering = (
                    "You are an AI assistant to designed to help people order boba tea drinks. Below is a list "
                    "of options, and some characteristics of each drink that you can use to recommend when appropriate.\n"
                    f"{get_descriptions_formatted(drink_list)}\n"
                    "If the user does not know what they want, help by asking leading questions to find which drink they "
                    "would like. Try to offer a maximum of 2 drinks, so they can make an easy choice.\n"
                    "Once they have decided on a drink, use the 'add_to_cart' tool to add it to the cart. "
                    "Do not offer adjustments or topping on the drink, and if the user asks inform them that this feature "
                    "is not yet implemented. Only one drink can be ordered at a time, and it must be listed on the menu. "
                    "Maintain a friendly, helpful customer service tone."
                    )

    # Once formatting has been completed, unwrap drinks from categories as those are not needed
    #  Also converts drinks to lowercase as that is how they are compared by the add_to_cart function
    drink_list = {drink.lower(): details for category in drink_list.values() for drink, details in category.items()}

    # Keep a list of conversation history
    messages = [{"role": "system", "content": sysprompt_ordering}]

    # Define tools for model
    tools = [
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "adds the passed drink to the cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "drink": {
                        "type": "string",
                        "description": "the name of the drink that should be added to the cart.",
                    }
                },
                "required": ["drink"],
            },
        },
    },
    ]

    co = cohere.ClientV2(api_key=api_key)

    selected_drink = None
    def add_to_cart(drink : str) -> bool:
        print(f"Added {drink} to your cart.")
        if drink.lower() in drink_list:
            nonlocal selected_drink
            selected_drink = drink
            return True
        # Not a valid drink, return False to signal this
        return False
    
    # Dict to support other functions as well, although only a single one is used here
    functions_map = {"add_to_cart" : add_to_cart}

    # Loop until drink has been selected
    while not selected_drink:
        input_text = input("You >> ")
        messages.append(
                {"role": "user", "content": input_text}
            )

        response = co.chat(
            messages=messages,
            model="command-a-03-2025",
            response_format={
                "type": "text"
                },
            tools=tools,
        )
        messages.append(response.message)
        
        if response.message.tool_calls:
            # If the model chose to add something to cart, do so here
            for tc in response.message.tool_calls:
                tool_result = functions_map[tc.function.name](
                    **json.loads(tc.function.arguments)
                )
                if tool_result == False:
                    # If the drink the model selected was not valid, ask the user to try again.
                    #  I actually wasn't able to trigger this in testing, but checking here is safer
                    print("Bot >> I'm sorry, I didn't quite understand that. Which drink would you like to order?")
                    # This will automatically loop until something has been added to the cart, so no special action
                    #  is needed.

        else:
            # Otherwise, continue conversation
            print("Bot >>", response.message.content[0].text)

    # Toppings/sweetness adjustment conversation would go here
    #  Change out the system prompt for toppings/sweetness/similar and repeat above process
    print("------------------------------")
    print(f"You ordered the drink: {selected_drink}")
    print("Enjoy!")


if __name__ == "__main__":
    order_drink()
