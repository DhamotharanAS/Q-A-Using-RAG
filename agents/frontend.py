
import streamlit as st
import requests

# Streamlit app
st.title("Q&A Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Question input
question = st.text_input("Enter your question")

# Button to submit
if st.button("Submit"):
    if uploaded_file is not None and question:
        # Convert the uploaded file to bytes
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
        data = {'question': question}
        
        # Make the request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/uploadfile/", files=files, data=data)
        
        # Check the response
        if response.status_code == 200:
            response_json = response.json()
            st.write("The Question is:", response_json.get("The Question is"))
            st.write("The Answer is:", response_json.get("The Answer is"))
        else:
            st.write("Error:", response.status_code)
    else:
        st.write("Please upload a PDF file and enter a question.")