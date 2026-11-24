import os

import streamlit as st
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HR Screening Agent", page_icon="🧑‍💼", layout="centered")

# ---------------------------------------------------------------------------
# API key: Streamlit secrets (deployed) -> environment variable -> sidebar input
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


api_key = get_api_key()

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
    else:
        st.success("API key loaded")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("Workflow: categorize experience → assess skills → route to interview / recruiter / reject")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str


# ---------------------------------------------------------------------------
# Build the LangGraph workflow (cached so it is not rebuilt on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_app(_api_key: str, model: str):
    llm = ChatOpenAI(model=model, temperature=0, api_key=_api_key)

    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the following job application, categorize the candidate as "
            "'Entry-level', 'Mid-level' or 'Senior-level'. "
            "Respond with only one of these three labels. "
            "Application : {application}"
        )
        level = (prompt | llm).invoke({"application": state["application"]}).content
        return {"experience_level": level.strip()}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the job application for a Python Developer, assess the candidate's skillset. "
            "Respond with only 'Match' or 'No Match'. "
            "Application : {application}"
        )
        skill = (prompt | llm).invoke({"application": state["application"]}).content
        return {"skill_match": skill.strip()}

    def schedule_hr_interview(state: State) -> State:
        return {"response": "Candidate has been shortlisted for an HR interview."}

    def escalate_to_recruiter(state: State) -> State:
        return {"response": "Candidate has senior-level experience but doesn't match job skills."}

    def reject_application(state: State) -> State:
        return {"response": "Candidate doesn't meet JD and has been rejected."}

    def route_app(state: State) -> str:
        skill = state["skill_match"].strip().lower()
        level = state["experience_level"].strip().lower()
        if skill.startswith("match"):
            return "schedule_hr_interview"
        elif "senior" in level:
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(State)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


def run_candidate_screening(app, application: str) -> dict:
    results = app.invoke({"application": application})
    return {
        "experience_level": results["experience_level"],
        "skill_match": results["skill_match"],
        "response": results["response"],
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧑‍💼 HR Candidate Screening Agent")
st.write("Paste a candidate's application or profile summary and the agent will screen it for a Python Developer role.")

EXAMPLES = {
    "— choose an example —": "",
    "Senior, Java only": "I have 10 years of experience in software engineering with expertise in JAVA",
    "Junior, Java only": "I have 1 year of experience in software engineering with expertise in JAVA",
    "Python developer": "I have 4 years of experience in software engineering with expertise in Python, Django and Flask",
}
choice = st.selectbox("Quick examples", list(EXAMPLES.keys()))

application_text = st.text_area(
    "Candidate application",
    value=EXAMPLES[choice],
    height=180,
    placeholder="e.g. I have 5 years of experience in backend development with Python and FastAPI...",
)

if st.button("Screen candidate", type="primary"):
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar or in Streamlit secrets.")
    elif not application_text.strip():
        st.warning("Please enter an application first.")
    else:
        try:
            with st.spinner("Screening candidate..."):
                app = build_app(api_key, model_name)
                result = run_candidate_screening(app, application_text.strip())

            st.subheader("Result")
            col1, col2 = st.columns(2)
            col1.metric("Experience level", result["experience_level"])
            col2.metric("Skill match", result["skill_match"])

            decision = result["response"]
            if "shortlisted" in decision:
                st.success(decision)
            elif "senior-level" in decision:
                st.warning(decision)
            else:
                st.error(decision)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
