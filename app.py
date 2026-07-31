
import streamlit as st
import pandas as pd
from rapidfuzz import process
import google.generativeai as genai


# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Pharma AI",
    page_icon="💊"
)



# ==========================
# Gemini Setup
# ==========================




model = genai.GenerativeModel(
    "models/gemini-3.1-flash-lite"
)


# ==========================
# Load Dataset
# ==========================

@st.cache_data
def load_data():

    url = "https://huggingface.co/mohamed22264/imputdata_pharma/resolve/main/data2_imputed.csv"

    df = pd.read_csv(url)

    df.columns = df.columns.str.strip()

    return df


df = load_data()



# ==========================
# Title
# ==========================

st.markdown(
    """
    <h1 style='text-align:center;color:#1565C0'>
    💊 Pharma AI Assistant
    </h1>
    """,
    unsafe_allow_html=True
)


st.write(
    "مساعد صيدلي ذكي للحصول على معلومات عامة عن الأدوية."
)



# ==========================
# Search Medicine
# ==========================

# ==========================
# Search Medicine
# ==========================

def search_medicine(question):

    names = df["Name"].astype(str).tolist()

    match = process.extractOne(
        question,
        names,
        score_cutoff=60
    )

    if not match:
        return None

    medicine = df[
        df["Name"] == match[0]
    ].iloc[0]

    info = f"""

Medicine Name:
{medicine['Name']}

Category:
{medicine['Category']}

Dosage Form:
{medicine['Dosage Form']}

Strength:
{medicine['Strength']}

Classification:
{medicine['Classification']}

Pregnancy Warning:
{medicine['Pregnancy_Warning']}

Contraindications:
{medicine['Contraindications']}

Side Effects:
{medicine['Side_Effects']}
"""

    return info

# ==========================
# Gemini Explanation
# ==========================

def ask_gemini(medicine_info):


    prompt = f"""

You are Pharma AI assistant.

Use ONLY the information below.

Do not add new medical facts.
Do not diagnose diseases.
Do not prescribe medicines.

Explain this medicine information in simple Arabic.

Medicine Information:

{medicine_info}

"""


    response = model.generate_content(
        prompt
    )


    return response.text



# ==========================
# Chat Memory
# ==========================

if "messages" not in st.session_state:

    st.session_state.messages = []



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])




# ==========================
# Chat Input
# ==========================

question = st.chat_input(
    "💬 اسأل عن دواء..."
)



if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.write(question)



    with st.chat_message("assistant"):


        with st.spinner("🤖 جاري البحث..."):


            medicine_info = search_medicine(
                question
            )


            if medicine_info:
                # لو الدواء موجود
                answer = ask_gemini(medicine_info)

            else:
                # لو الدواء مش موجود
                prompt = f"""
You are Pharma AI assistant.

The medicine was not found in the local database.

If you know the medicine from your medical knowledge, answer about it.

If you do not know the medicine, politely tell the user that you do not have reliable information.

Rules:
- Answer in Arabic.
- Explain the medicine uses.
- Mention common side effects.
- Mention pregnancy warnings if known.
- Do not diagnose diseases.
- Do not prescribe medicines.
- Keep the answer short.

User question:
{question}
"""

                try:

                    response = model.generate_content(prompt)

                    answer = response.text

                except Exception:

                    answer = """
❌ لم أجد الدواء في قاعدة البيانات المحلية، وتعذر الاتصال بنموذج الذكاء الاصطناعي.
"""


            st.markdown(answer)



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )



# ==========================
# Sidebar
# ==========================

with st.sidebar:


    st.title("💊 Pharma AI")


    st.write(
        """
يساعدك في:

✅ معلومات الأدوية

✅ الاستخدامات

✅ الأعراض الجانبية

✅ التحذيرات

⚠️ لا يغني عن الطبيب
"""
    )


    if st.button("🗑️ مسح المحادثة"):

        st.session_state.messages = []

        st.rerun()
