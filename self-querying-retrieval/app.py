from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# should be separate file
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

documents = [
    Document(
        page_content="Local supermarket",
        metadata={
            "name": "Billa",
            "category": "infrastructure",
            "subcategory": "supermarket",
            "address": "Hauptstraße 34/369, 2651 Reichenau an der Rax",
            "opening_hours": "7am",
            "closing_hours": "7:15pm",
            "service_options": "Delivery, In-store pick-up, In-store shopping",
            "accessibility": "Wheelchair-accessible car park, Wheelchair-accessible entrance",
            "phone": "05991503445",
            "payment_types": "Cash, Credit cards, Debit cards, NFC mobile payments",
            "new_offers": "All cheese at half-price for the entire month of December 2023.",
            "reviews": "Good Supermarkt With cheaps prices in AUSTRIA",
            "website_url": "https://www.billa.at/"
        }
    ),
    Document(
        page_content="",
        metadata={"name": "Evangelische Henriettenkapelle", "category": "touristic", "subcategory": "monument"}
    ),
    Document(
        page_content="",
        metadata={"name": "Gasthotel Kobald", "category": "hotel"}
    ),
    Document(
        page_content = "accommodation, Place to sleep, ",
        metadata = {"name":"Haus Trautenberg","category":"hotel"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"L'attore Ristorante e Pizzeria","category":"restaurant", "subcategory":"Italian"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"Parkhotel Hirschwang","category":"hotel"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"Parking 1","category":"infrastructure", "subcategory" : "parking"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"Payerbach-Reichenau","category":"infrastructure", "subcategory" : "train station"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"Schneeberg","category":"touristic", "subcategory" : "ski slope"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"Schwarzer Weg","category":"touristic", "subcategory" : "trekking trail"}
    ),
    Document(
        page_content = "",
        metadata = {"name":"bp","category":"infrastructure", "subcategory" : "gas station"}
    ),
]
document_content_description = "Brief description of local touristic points"

vectorstore = Chroma.from_documents(documents, embeddings)


metadata_field_info = [
    AttributeInfo(
        name = "name",
        description = "Name of the touristic point",
        type = "string or list [string]"
    ),
    AttributeInfo(
        name = "category",
        description = "The category of the touristic point",
        type = "string or list [string]"
    ),
    AttributeInfo(
        name = "subcategory",
        description = "The subcategory of the touristic point",
        type = "string or list [string]"
    ),
]

llm = OpenAI(temperature = 0)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose = True
)

# examples

test = retriever.get_relevant_documents("What are some hotels near by?")
print(test)