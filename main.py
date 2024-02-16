from functions import *
from datetime import datetime
import chainlit as cl
from ragas import evaluate 
from datasets import Dataset
from ragas.metrics import ( 
        faithfulness, 
        answer_relevancy, 
        context_precision, 
        # context_recall, 
    ) 
from ragas.metrics.critique import harmfulness 
from ragas.langchain.evalchain import RagasEvaluatorChain 
from ragas.metrics import ( 
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall, 
) 
 

import os
import openai
# Set OPENAI_API_KEY in environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]



def bot():
    llm = get_LLM()
    # st.session_state["llm"] = llm
    db = get_embeddings()
    prompt = get_prompt()
    chain = get_chain(llm, prompt, db)
    return chain,db

  

class Flag:
    flag=1

fl=Flag()

if(fl.flag):
    chain,db = bot()
    retriever = db.as_retriever()
    fl.flag = 0
    faithfulness_chain = RagasEvaluatorChain(metric=faithfulness) 
    answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy) 
    context_rel_chain = RagasEvaluatorChain(metric=context_precision) 
    context_recall_chain = RagasEvaluatorChain(metric=context_recall)
@cl.on_chat_start
async def start():
    
    
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello, Welcome to  StockBot. What is your query?"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    response = await chain.ainvoke(message.content, callbacks=[cb])
    print(response)

    result = ""
    eval_result = faithfulness_chain.invoke(response) 
    result = result+"Faithfulness: "+str(eval_result["faithfulness_score"])
    
    
    eval_result = answer_rel_chain.invoke(response) 
    result = result+"\nAnswer Relevancy:"+str(eval_result["answer_relevancy_score"])
    
    eval_result = context_rel_chain.invoke(response) 
    result = result+"\nContext Precision: "+str(eval_result["context_precision_score"])
    answer = response["result"]

    # answer = answer+"\n\nSCORE"+result
    print("\n\n\nSCORE\n",result,"\n\n")

    await cl.Message(content=answer).send()
