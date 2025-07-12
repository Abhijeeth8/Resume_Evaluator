from dotenv import load_dotenv
load_dotenv()

import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.tools import WriteFileTool
import streamlit as st
import pypdf



llm = ChatOpenAI(temperature=1, model="gpt-4o")
llm_with_tools = llm.bind_tools([WriteFileTool()])

tools_dict = {}
tools_dict[WriteFileTool().get_name()] = WriteFileTool()

def load_pdf_and_md_it(pdf_path="./Abhijeeth_Kollarapu_ds_ml_ai_all_projects.pdf"):
  pdf_loader = PyPDFLoader(pdf_path)
  resume_doc = pdf_loader.load()
  full_resume = '\n'. join([res.page_content for res in resume_doc])

  resume_md_creator_prompt_template = PromptTemplate(template="You are a resume converter that converts the given resume text to markdown format without changing anything in it and stores it in a file with the name resume_markdown.md. Make sure to convert to markdown. You can use any tools availabe to you to achieve the task.\n Resume : {resume}",
  input_variables=["resume"], validate_template=True)

  resume_md_creator_prompt = resume_md_creator_prompt_template.invoke(input={"resume":full_resume})
  res = llm_with_tools.invoke(resume_md_creator_prompt)
  md_ready = tools_dict[res.tool_calls[0]['name']].invoke(res.tool_calls[0])
  return res.tool_calls[0]['args']['file_path']

def load_md_resume(resume_md_file_path):
  with open(resume_md_file_path, 'rb') as f:
    md_resume_bytes = f.read()
  md_resume = md_resume_bytes.decode('utf-8')
  return md_resume

def load_pdf_resume(pdf_path="./Abhijeeth_Kollarapu_ds_ml_ai_all_projects.pdf"):
  pdf_loader = PyPDFLoader(pdf_path)
  resume_doc = pdf_loader.load()
  full_resume = '\n'. join([res.page_content for res in resume_doc])
  return full_resume

def get_jd(jd_file_path = "jd_file.txt"):
  with open(jd_file_path, 'r') as f:
    jd = f.read()

  return jd

def main(jd, resume_content):

  # resume_md_file_path = load_pdf_and_md_it(pdf_path)

  # md_resume = load_md_resume(res_md_file_path)
  md_resume = resume_content

  res_jd_eval = PromptTemplate(
    template=
  """
  You are a professional job-to-resume alignment assistant.\n
  Your purpose is to evaluate how well a resume is and how well it aligns with a given job description, identify the top matching projects, and suggest targeted improvements — without missing any required component and ignore the soft skills and only focus on the technical skills.\n
  \n
  ROLE:\n
  As an expert in resume screening, ATS evaluation, and job-fit analysis, your responsibilities include:\n
  - First greet the person who uploaded the resume.
  - Evaluating how good the resume is and how impactful the keywords and overall resume is and give the evaluation score and improvements.\n
  - Assessing resume alignment with the job description\n
  - Extracting technical skills, work experience and education details from the job description\n
  - Extracting and optimizing the top 3 most relevant projects\n
  - Highlighting matching/missing technical skills\n
  - Delivering actionable improvements\n
  \n
  INPUTS:\n
  You will be provided with:\n
  - resume in Markdown format\n
  - job description as plain text\n
  \n
  OUTPUT:\n
  You must generate a Markdown that includes:\n
  Greeting the user
  1. Score of how good and effective the resume is in terms of keywords.\n
  2. Match Score (out of 100) and why is it the score like on what basis it is being calculated\n
    - Clearly displayed at the top of the page.\n
  3. Top 3 Most Relevant Projects from the Resume\n
    - Select exactly three projects most aligned with the job description.\n
    - Within each selected project:\n
      - Highlight weak, vague, or ineffective parts and suggest improvements if any.\n
  4. Technical Skills Summary Section\n
    Extract all the required technical skills from the job description as a list and make sure to not miss even a single required skill. And then tick which are present in the resume and which are not.
  5. General Improvements Section (if applicable)\n
    - Provide any additional suggestions to improve resume alignment or clarity.\n
  \n
  IMPORTANT:\n
  - Do not skip or omit any required section.\n
  - Ensure the output is complete and well-structured in Markdown without any backticks or anything else other than Markdown.\n
  - You don't have to save it anywhere just output the response.\n
  \n

  Resume: {resume}\n
  Job Description: {jd}""",

  input_variables=["resume", "jd"], validate_template=True
  )

  # jd = input("Paste Job Description here:\n")

  # jd_file_path = ""
  # if(jd_file_path):
  #   jd = get_jd(jd_file_path)
  # else:
  #   jd = get_jd()

  res_jd_eval_prompt = res_jd_eval.invoke(input={"resume":md_resume, "jd":jd})

  evaluation_report_ai_response = llm_with_tools.invoke(res_jd_eval_prompt)
  evaluation_report = evaluation_report_ai_response.content

  # evaluation_report_file_path = "evaluation_report.md"

  # with open(evaluation_report_file_path, 'w') as f:
  #   f.write(evaluation_report)

  return evaluation_report

def generate_cover_letter(jd, resume_content ):

  # md_resume = load_md_resume(res_md_file_path)
  md_resume = resume_content

  cover_letter_gen_prompt_template  =PromptTemplate(
    template=
    """
You are an expert career coach and professional resume writer.

Generate a complete, ready-to-send cover letter for the role described below.

Instructions:
- Extract the applicant’s full name and contact details (email, phone, LinkedIn, GitHub) from the resume.
- Extract the job title, company name, and location from the job description.
- Begin the letter with a plain-text header showing the applicant’s name and contact details each on its own line.
- Below that, include today’s date in a simple format (e.g., July 7, 2025).
- Then add the hiring manager’s title (“Hiring Manager”), company name, and company location on separate lines.
- Do not use any LaTeX, markdown, backticks, or special formatting—just plain text.
- Write a professional, concise cover letter body tailored to the job.
- Highlight the most relevant skills and achievements from the resume that align with the job description.
- Use specific examples and metrics where appropriate.
- End with a confident closing and a call to action.
- Keep the entire letter under 400 words.
- The output should be only the complete plain-text cover letter, ready to send.

Inputs:
Resume:
{resume}

Job Description:
{jd}
""", input_variables = ["resume", "jd"], validate_template=True
  )

  cover_letter_gen_prompt = cover_letter_gen_prompt_template.invoke(input={"resume":md_resume, "jd":jd})
  cover_letter = llm_with_tools.invoke(cover_letter_gen_prompt)
  return cover_letter.content

def why_this_company(jd, resume_content):
  # md_resume = load_md_resume(res_md_file_path)
  md_resume = resume_content

  why_this_comp_prompt_template  =PromptTemplate(
    template=
    """
You are an expert career coach and professional writer.

Using the provided resume and job description, write a thoughtful and genuine answer to the question:

“Why are you interested in this particular company?”

Instructions:
- Extract relevant details about the candidate’s skills, experiences, and values from the resume.
- Extract information about the company’s mission, values, culture, products, or recent news from the job description.
- Write a personalized and specific response showing alignment between the candidate’s background and the company’s goals.
- Use a professional, enthusiastic tone.
- Avoid generic or vague statements.
- Keep the answer concise (about 150-200 words).
- Provide the answer only in plain text, no formatting or extra text.

Inputs:
Resume:
{resume}

Job Description:
{jd}
""", input_variables = ["resume", "jd"], validate_template=True
  )

  why_this_comp_prompt = why_this_comp_prompt_template.invoke(input={"resume":md_resume, "jd":jd})
  answer = llm_with_tools.invoke(why_this_comp_prompt)
  return answer.content



if __name__ == "__main__":
  # evaluation_report = main()

  # resume_url = "Abhijeeth_Kollarapu_ds_ml_ai_all_projects.pdf"
  # file_type  = resume_url.split('.')[-1]
  # if(file_type == "md"):
  #   resume_content = load_md_resume(resume_url)
  # elif(file_type == "pdf"):
  #   resume_content = load_pdf_resume(resume_url)

  # print(resume_content)

  st.set_page_config(page_title="Resume-JD Evaluator", layout="wide")

  st.title("🔍 Resume Optimizer")

  st.markdown("Upload your **resume (PDF or md)** and a **job description (text)** to get a detailed markdown evaluation.")

  resume_file = st.file_uploader("Upload Resume pdf or md", type=["pdf", "md"])
  jd_text = st.text_area("Paste Job Description (Required)", height=300, placeholder="Paste job description here...")
  resume_content = ""

  if resume_file and jd_text:
      with st.spinner("Processing resume..."):

        file_type  = resume_file.type

        if(file_type == "application/pdf"):
          pdf_reader = pypdf.PdfReader(resume_file)
          resume_content = ""
          for page in pdf_reader.pages:
            resume_content += page.extract_text()
          
        else:
          resume_content = resume_file.read().decode('utf-8')
  # res_md_file_path  = "resume_markdown.md"
  if jd_text.strip():
    
    with st.spinner("Reading JD..."):
      jd = jd_text.strip()

    with st.spinner("Running evaluation..."):
      # md_resume = load_md_resume(res_md_file_path)
      # report = main(jd_text, res_md_file_path)
      report = main(jd_text, resume_content)

    st.success("✅ Evaluation Complete!")
    st.markdown("### 📄 Evaluation Report")
    st.markdown(report, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("✉️ Generate Personalized Answer to Why this company")

    if st.button("Generate Personalized answer"):
        with st.spinner("generating personalized answer..."):
            answer = why_this_company(jd, resume_content)

        st.success("✅ Answer generated!")
        st.markdown("### 📄 Answer")
        st.text_area("Generated Answer", value=answer, height=300)


        # -------- Cover Letter Generation Section -------- #
    st.markdown("---")
    st.subheader("✉️ Generate Personalized Cover Letter")

    if st.button("Generate Cover Letter"):
        with st.spinner("Creating cover letter..."):
            cover_letter = generate_cover_letter(jd, resume_content)

        st.success("✅ Cover letter generated!")
        st.markdown("### 📄 Cover Letter")
        st.text_area("Generated Cover Letter", value=cover_letter, height=300)


        # st.download_button(
        #     "💾 Download Cover Letter",
        #     data=cover_letter,
        #     file_name="cover_letter.pdf",
        #     mime="application/pdf"
        # )

  else:
      st.warning("Please upload both the Resume and Job Description text.")
