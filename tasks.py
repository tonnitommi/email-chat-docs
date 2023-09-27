from robocorp.tasks import task
from robocorp import workitems, vault
from RPA.Notifier import Notifier

import openai

from llama_index import VectorStoreIndex, SimpleDirectoryReader

import os

DATA_FOLDER = "data"
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

    print(f"------Iterating the questions ------")

    body = "Replies to your questions:\n\n"

    for line in response['choices'][0]['message']['content'].splitlines():
        response = query_engine.query(line)
        body = body + f"Question: {line}\n"
        body = body + f"Response: {response}\n\n"

    notifier = Notifier()
    gmail_credentials = vault.get_secret("Google")

    notifier.notify_gmail(body, email.from_.address, gmail_credentials["email"], gmail_credentials["email-app-password"])


def create_index(folder):
    """Creates the vector index out of all files"""

    reader = SimpleDirectoryReader(folder)
    docs = reader.load_data()

    index = VectorStoreIndex.from_documents(docs)
    
    return index

