from robocorp.tasks import task
from robocorp import workitems, vault
from RPA.Notifier import Notifier

from llama_index import VectorStoreIndex, SimpleDirectoryReader

import openai
import os

# TODO: Make code prettier, maybe in some smaller methods
# TODO: Check for too many tokens in prompts

DATA_FOLDER = "data"
THRESHOLD = 0.6
QUESTIONS_EXTRACT_PROMPT = """Your task is to extract the users questions, and questions only, from
the following email body. User is asking questions regarding documents, and the extracted
questions will be used in the next step one by one.

Return the questions individually each on it's own line, and do not add any extra characters
or explanations to your reply. Retain user's original as much as possible.

If there is no questions about the documents in the email, simply return one line with text NONE and
nothing else.

Follow the example below.

*** Example starts ***
<User's message>
Hi AI, I would like to know some details of the documents attached. What is the year of the
document? Then also tell me what was the revenue reported in the year in question. Was it
growing compared to last year?

Thanks,
Some person

<Your output>
What is the year of the document?
What was the revenue reported in the year in question?
Was the revenue growing compared to previous year?
*** Example ends ***

Now your turn, extract questions from this message:

"""

FINAL_PROMPT_TEMPLATE = f"""Your task is to answer the following question:\n\n{line}\n\n
Use only the information provided by the following contextual information that another
AI assistant has extracted from the document provided by the user. If the contextual
information does not provide an answer to the question, clearly state that. Do not rely
on your generic information in answering the question. You may use tables and bullet list
to make the information easily understanable, if they make sense in the context of the
question.

Along with the context, also the source is mentioned (the document and the page). In your
final response, include the sources either in the relevant places of your response, or in the
end.
"""

@task
def chat_with_docs():
    """Read docs from input work item and answer questions."""

    openai_credentials = vault.get_secret("OpenAI")
    os.environ["OPENAI_API_KEY"] = openai_credentials["key"]
    openai.api_key = openai_credentials["key"]

    # Get the input work item and try getting the email and
    # .pdf attachments out of it.
    item = workitems.inputs.current
    try:
        email = item.email()
        paths = item.get_files("*.pdf", DATA_FOLDER)
    except Exception as e:
        print("Problem with emails:", str(e))
        return

    # If no files are found, no reason to continue.
    if not paths:
        print("No files, exiting")
        return

    # Get the questions out of the full email body using LLM.
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Your are an assistant helping extract structured data from the messages."},
            {"role": "user", "content": QUESTIONS_EXTRACT_PROMPT + email.text}
        ]
    )

    # Check for NONE response
    if response['choices'][0]['message']['content'] == "NONE":
        print("User's email body did not contain any questions. Exiting.")
        return

    index = create_index(DATA_FOLDER)
    query_engine = index.as_query_engine()

    body = f"Hi {email.from_.name}!\n\nHere are the replies to your questions:\n\n"

    # Iterate over all the questions
    for line in response['choices'][0]['message']['content'].splitlines():

        # The first query looks for the relevant contexts.
        # TODO: Add more info to the prompt, now it's just a basic question
        # without any instructions.
        response = query_engine.query(line)

        # TODO: Put this to templates


        # For each found context or "node" add them to prompt if their score is high enough.
        for node in response.source_nodes:

            if node.score > THRESHOLD:
                final_prompt = final_prompt + "\n\n***CONTEXT***\n" + node.text + f"\n***SOURCE:*** File: {node.metadata['file_name']}, page{node.metadata['page_label']}\n"
            else:
                print("Score too low, ignoring the node.")

        # Do the final question
        final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Your are an assistant helping to answer user's question based on the information found by other AI Assistants."},
                {"role": "user", "content": final_prompt}
            ]
        )

        body = body + "-------------------------------------------------------\n\n"
        body = body + f"Question: {line}\n\n"
        body = body + f"Response: {final_response['choices'][0]['message']['content']}\n\n"

    # This is outside of the for loop
    notifier = Notifier()
    gmail_credentials = vault.get_secret("Google")

    # TODO: Change to use better library, and make it a reply...
    # TODO: Make them html emails
    notifier.notify_gmail(
        message=body,
        to=email.from_.address,
        username=gmail_credentials["email"],
        password=gmail_credentials["email-app-password"],
        subject="Message from a friendly bot")


def create_index(folder):
    """Creates the vector index out of all files"""

    reader = SimpleDirectoryReader(folder)
    docs = reader.load_data()

    index = VectorStoreIndex.from_documents(docs)

    return index

