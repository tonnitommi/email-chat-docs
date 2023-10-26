# The simple RAG with Llamaindex, OpenAI and Robocorp tutorial.

Build RAG in an hour! This tutorial walks throught he build of a simple RAG agent using [OpenAI](https://openai.com/), [Llamaindex](https://www.llamaindex.ai/) and running it on Robocorp. Here's what it will do:

> Hey I have these questions about these documents. Help me!

- Workflow is triggered via emails where body has some questions about the PDF files as attachments. The email triggering is a great feature on Robocorp Control Room, and is available for every Python workflow.
- Looks at email body, and extracts clear questions out of it.
- Creates embeddings and the vector index on the fly from the documents found from attachments.
- Finds relevant contexts or "nodes" from the index
- Feeds them to the final step for answering the original questions.
- Sends the results back as an email.


## Get your setup ready

Robocorp is a great way of running these types of Python workloads, with zero infrastructure setup needed!

Still, a few things needs to be in place for things to work. Follow these steps and you are good to go!

1. Make sure you have [VS Code](https://code.visualstudio.com/download) installed, and the [Robocorp Code extension](https://marketplace.visualstudio.com/items?itemName=robocorp.robocorp-code) activated. Then clone the [repo](https://github.com/tonnitommi/email-chat-docs).
2. Robocorp creates fully isolated Python environments based on the [conda.yaml](conda.yaml) configuration files, so just remember no `pip install` of anything needed.
3. Some credentials are needed for things to run: OpenAI and Gmail (optional). You can have them in the Robocorp Control Room and connect the VS Code to your workspace, but here we assume that you are not there yet. We'll use the local secrets file for now.
   1. Remove the `_EXAMPLE` from the name of the [secrets.json_EXAMPLE](secrets.json_EXAMPLE) file.
   2. Modify the file to have your own OpenAI API key.
   3. (OPTIONAL) If you want the automation to send replies as emails, add your own gmail address, and app password. [Here's how to generate the password.](https://support.google.com/mail/answer/185833?hl=en). If you don't want to do that, the workflow is just printing the final responses to the console.
   4. (OPTIONAL) If you chose to add your gmail credentials, then also edit the example work item that you can use for testing locally. Look for from/email address in the [work-items.json](devdata/work-items-in/example-email-big/work-items.json) file. We'll show later how you can run things locally to test!

With these done, you are ready to go!

## Step by step

code examples

## Running it

Run locally

Deploy to cloud



