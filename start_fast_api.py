from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from chatbot.chatbot import (setup_vector_store, setup_embedding_model, get_output_prompt_for_one_employee,
                             get_summary_chain, get_summary, get_output, get_personal_ids_for_query,
                             get_image_description)
import os
from langchain_openai import OpenAI
import requests
import base64


class BotInput(BaseModel):
    input_image: str = None
    input_text: str = "Wer kann mir beim Thema IT Security helfen?"
    exclude_ids: list[int] = []


class VectorStoreUpdateInput(BaseModel):
    id: int  # d for database person that should be updated in vector store


# Set OpenAI API key to env variable OPENAI_API_KEY
# Set your database token to env variable DB_TOKEN="database_usage_token"

db_url = "https://gpt.hansehart.de/api/service"

app = FastAPI()
llm = OpenAI()
vision_model = OpenAI(model="gpt-4o")
embedding_model = setup_embedding_model()
vector_store = setup_vector_store(embedding_model, f"{db_url}/receive/persons", os.environ["DB_TOKEN"])
summary_chain = get_summary_chain(llm=llm)
output_prompt = get_output_prompt_for_one_employee()


@app.get("/")
async def root():
    return {"status": "Server is up and running"}


@app.post("/ask_bot/")
async def ask_bot(input_query: BotInput):
    if input_query.input_image:
        try:
            image_data = base64.b64decode(input_query.input_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 encoded image.")

        # Call OpenAI GPT-4 model with vision capabilities
        query = input_query.input_text
        result, personal_ids = process_image_query(query, image_data, vector_store, summary_chain, output_prompt, input_query.exclude_ids)
        return {"result_text": result, "personal_ids": personal_ids}
    else:
        result_text, personal_ids = process_query(input_query.input_text, vector_store, summary_chain, output_prompt, input_query.exclude_ids)
        return {"result_text": result_text, "personal_ids": personal_ids}


@app.post("/update_vector_store/")
async def update_vector_store(update_input: VectorStoreUpdateInput):
    try:
        print(update_input.id)

        # Trigger the update process (e.g., reload documents, re-index, etc.)
        pass
        return {"status": "Vector store update triggered successfully"}
    except Exception as e:
        return {"status": "Failed to trigger update", "error": str(e)}


def process_query(query, vector_store, summary_chain, output_prompt, exclude_ids):
    outputs = []
    personal_ids = get_personal_ids_for_query(query, vector_store, exclude_ids)
    if len(personal_ids) == 0 and len(exclude_ids) == 0:
        return ("Damit kann ich dir leider nicht weiterhelfen. Stelle, eine Frage im Bezug zu Personen unseres"
                + " Unternehmens."), []
    elif len(personal_ids) == 0 and len(exclude_ids) > 0:
        return ("Ich habe leider keine weiteren Ergebnisse gefunden."), []
    for personal_id in personal_ids:
        employee_data = requests.get(f"{db_url}/receive/person?id={personal_id}",
                                     headers={"Authorization": f"Bearer {os.environ['DB_TOKEN']}"}).json()
        summary = get_summary(summary_chain, query, employee_data)
        output = get_output(output_prompt, employee_data, summary)
        outputs.append(output)
    final_output = ("Die folgenden Mitarbeiter können dir behilflich sein:\n\n" + "\n\n".join(outputs))
    return final_output, personal_ids


def process_image_query(query, image_data, vector_store, summary_chain, output_prompt, exclude_ids):
    image_description = get_image_description(image_data)
    if not query:
        query = "Welche Person kann mir bei dem beschriebenen Sachverhalt helfen?"
    return process_query(image_description + " " + query, vector_store, summary_chain, output_prompt, exclude_ids)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
