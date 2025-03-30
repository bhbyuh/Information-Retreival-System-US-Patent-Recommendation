from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import cohere
load_dotenv()

# Initialization of cohere key 
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

# Initialization of OpenAI
openai_key=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model=os.getenv("OPENAI_MODEL"),temperature=0)
embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING"),
    api_key=openai_key
)

# Pinecone initialization
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def reranker(query):

    query_response = index.query(
                    vector=embeddings.embed_query(query),
                    top_k=25,
                    include_metadata=True
                    )
    
    docs=[]
    class_names=[]
    term_id=[]
    main_class_id=[]

    for i in range(len(query_response.get("matches"))):
        docs.append(query_response.get("matches")[i]['metadata']['context'])
        class_names.append(query_response.get("matches")[i]['metadata']['Class Name'])
        term_id.append(query_response.get("matches")[i]['metadata']['Term ID'])
        main_class_id.append(query_response.get("matches")[i]['metadata']['Class'])

    for indexx,term in enumerate(term_id):
        term_id[indexx]=int(term.split('-')[1])

    results = co.rerank(
        model=os.getenv("COHERE_MODEL"),
        query=query,
        documents=docs,
        top_n=5)
    
    rank_docs=[]
    rank_class_names=[]
    rank_term_id=[]
    rank_main_class_id=[]

    results=results.results

    for a in results:
        rank_docs.append(docs[a.index])
        rank_class_names.append(class_names[a.index])
        rank_term_id.append(term_id[a.index])
        rank_main_class_id.append(main_class_id[a.index])

    # Sort on base of similar Main class
    combined=zip(rank_docs,rank_class_names,rank_term_id,rank_main_class_id)

    sorted_combined=sorted(combined,key=lambda x:x[3])

    docs,class_names,term_id,main_class_id=map(list, zip(*sorted_combined))
    
    output={}

    #extract unique classes
    seen = set()
    unique_classes = []
    unique_classes_id = []

    for classs, val in zip(class_names, main_class_id):
        if classs not in seen:
            seen.add(classs)
            unique_classes.append(classs)
            unique_classes_id.append(val)

    #make required payload structure
    for classs,idd in zip(unique_classes,unique_classes_id):
        output.update({f"{classs}":{"class id":idd,"sub classes":[]}})
    
    for doc,termm_id,class_name in zip(docs,term_id,class_names):
        for uniq_class in unique_classes:
            if uniq_class==class_name:
                output[class_name]["sub classes"].append({"sub class id":termm_id,"description":doc})
    
    return output