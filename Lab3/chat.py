from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st;
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature = 0)
st.title("movie summary generator")
user_input=st.text_input("Enter the name of the movie")
if st.button("Generate Summary"):
    response = model.invoke(user_input)
    st.write(response.content)


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview", 
#     temperature=0
# )

# query = "Tell me a joke on programming"
# response = llm.invoke(query)
# print(response.content)
