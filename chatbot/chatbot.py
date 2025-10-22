import os
import langchain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import requests
from openai import OpenAI

db_url = "https://gpt.hansehart.de/api/service"
imgbb_url = "https://api.imgbb.com/1/upload"

def setup_embedding_model():
    model = "text-embedding-3-large"
    embedding_model = OpenAIEmbeddings(model=model)
    return embedding_model


def setup_vector_store(embedding_model, db_url, db_token):
    database = requests.get(db_url, headers={"Authorization": f"Bearer {db_token}"}).json()
    documents = create_documents_from_db(database)

    """
    path_to_xlsx = "KI-HackathonxROSSMANN_Challenge_Ansprechpartner-Chatbot.xlsx"
    df = pd.read_excel(path_to_xlsx)
    documents = create_documents_from_df(df)
    """

    vector_store = Chroma(embedding_function=embedding_model)
    vector_store.add_documents(documents=documents)
    return vector_store


def get_personal_ids_for_query(query, vector_store, exclude_ids):
    #documents = vector_store.similarity_search(query.input_text, k=3)
    documents = []

    if not isinstance(query, str):
        query = query.input_text
    if query == "":
        return []
    docs, scores = zip(*vector_store.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        if score < 1.5 and doc.metadata.get("personal_id") not in exclude_ids:
            documents.append(doc)
        if len(documents) == 3:
            break

    ids = [document.metadata.get("personal_id") for document in documents]
    ids = list(dict.fromkeys(ids))
    return ids


def create_documents_from_db(database):
    documents = []
    for employee in database:
        job_description = employee.get("beschreibung") if employee.get("beschreibung") else ""
        programs = employee.get("programme") if employee.get("programme") else ""
        personal_id = employee.get("id")

        description_document = Document(page_content=job_description, metadata={"personal_id": personal_id})
        job_document = Document(page_content=programs, metadata={"personal_id": personal_id})

        documents.append(job_document)
        documents.append(description_document)
    return documents


def create_documents_from_df(df):
    documents = []
    for index, row in df.iterrows():
        document = Document(page_content=row["Beschreibung der Position und Zuständigkeiten bei Problemen"], metadata={"personal_id": index})
        documents.append(document)
        document = Document(page_content=row.fillna("")["Betreute Programme"], metadata={"personal_id": index})
        documents.append(document)
    return documents


def delete_employee_from_vector_store(personal_id, vector_store):
    document_ids = vector_store.get(where={"personal_id": personal_id}.get("ids"))
    vector_store.delete(ids=document_ids)
    return vector_store


def add_employee_to_vector_store(personal_id, vector_store):
    employee = requests.get(f"{db_url}/receive/person?id={personal_id}",
                            headers={"Authorization": f"Bearer {os.environ['DB_TOKEN']}"}).json()

    job_description = employee.get("beschreibung") if employee.get("beschreibung") else ""
    programs = employee.get("programme") if employee.get("programme") else ""
    personal_id = employee.get("id")

    description_document = Document(page_content=job_description, metadata={"personal_id": personal_id})
    job_document = Document(page_content=programs, metadata={"personal_id": personal_id})
    vector_store.add_documents(documents=description_document)
    vector_store.add_documents(documents=job_document)


def update_employee_by_personal_id(personal_id, vector_store):
    delete_employee_from_vector_store(personal_id, vector_store)
    add_employee_to_vector_store(personal_id, vector_store)
    return vector_store


def get_summary_chain(llm):
    summary_template = """
    Erstelle eine kurze Zusammenfassung, warum der Mitarbeiter bei der Anfrage helfen kann. Schreibe nur einen Satz,
    maximal zwei. Der Satz soll grammatisch korrekt und vollständig sein. Gehe eher darauf ein, was der Mitarbeiter
    macht und von welchen Themen er Kenntnisse hat. Wiederhole nicht, auch nicht umschrieben, die Anfrage.
    Nenne den Mitarbeiter nur beim Vornamen. Entferne alle Zeilenumbrüche und vorangestellte Leerzeichen.

    Anfrage: {query}
    Vorname: {first_name}
    Nachname: {last_name}
    Job: {job}
    Beschreibung des Jobs: {job_description}

    Diese Person kann helfen, weil:
    """
    summary_prompt = PromptTemplate(
        input_variables=["query", "first_name", "last_name", "job", "job_description"],
        template=summary_template
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    return summary_chain


def get_summary(summary_chain, query, employee_data):
    summary = summary_chain.invoke(input={"query": query,
                                          "first_name": employee_data.get("vorname"),
                                          "last_name": employee_data.get("nachname"),
                                          "job": employee_data.get("position"),
                                          "mail": employee_data.get("mail"),
                                          "telefon": employee_data.get("telefon"),
                                          "job_description": employee_data.get("beschreibung")
                                          }
                                   )
    return summary


def get_output_prompt_for_one_employee():
    template = "{first_name} {last_name}, {job}\nMail: {mail}\nTelefon: {phone}\n{summary}"""
    output_prompt = PromptTemplate(
        input_variables=["first_name", "last_name", "job", "mail", "phone", "summary"],
        template=template
    )
    return output_prompt


def get_output(output_prompt, employee_data, summary):
    # Benutze die generierte Zusammenfassung im Prompt
    output = output_prompt.format(
        first_name=employee_data.get("vorname"),
        last_name=employee_data.get("nachname"),
        job=employee_data.get("position"),
        mail=employee_data.get("mail"),
        phone=employee_data.get("telefon"),
        summary=summary["text"].replace("\n", "")
    )
    return output


def get_image_description(image_data):
    vision_model = OpenAI()
    image_response = requests.post(
        "https://api.imgbb.com/1/upload",
        files={'image': image_data},
        data={'key': os.environ["IMGBB_KEY"], "expiration": 120}
    )
    image_url = image_response.json()["data"]["url"]
    #image_url = "https://as1.ftcdn.net/v2/jpg/05/77/57/36/1000_F_577573624_giM5eaasMgUxbuyB2QanQcchnqBkzEb0.jpg"

    response = vision_model.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
             }
        ]
    )
    return response.choices[0].message.content



