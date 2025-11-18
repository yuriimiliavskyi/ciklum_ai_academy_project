from agentProfessor import create_chatbot
from langsmith import Client
from langchain_openai import ChatOpenAI
from langsmith import evaluate
from langchain_classic.evaluation import load_evaluator
import os
from dotenv import load_dotenv

load_dotenv()
OpenAI_API = os.getenv("OpenAI_API")
LangChain_API = os.getenv("LangChain_API")

os.environ["OPENAI_API_KEY"] = OpenAI_API
os.environ["LANGCHAIN_API_KEY"] = LangChain_API
os.environ["LANGCHAIN_PROJECT"]="My Agent Evaluation" ## if needed, not sure
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://eu.api.smith.langchain.com"

# # Create the dataset - only once, then read
# dataset_description = "Evaluating lecture notes for relevance and clarity."
# dataset = client.create_dataset(
#     dataset_name=dataset_name,
#     description=dataset_description,
# )

# # Add examples
# client.create_example(
#     dataset_id=dataset.id,
#     inputs={"input": "Describe some challenging open problems of agentic AI",
#             "output":""}
# )

# client.create_example(
#     dataset_id=dataset.id,
#     inputs={"input": "Speak about 3 ethical risks of Agentic AI systems",
#             "output":""}
# )

# client.create_example(
#     dataset_id=dataset.id,
#     inputs={"input": "Outline main areas of application of Agentic AI",
#             "output":""}
# )

chatbot = create_chatbot(MY_PDF_FILE="../Agentic_AI_A_Comprehensive_Survey_of_Technologies_.pdf"
                         , RECIPIENT_EMAIL="ymil@ciklum.com")

def evaluate_chatbot():
    client = Client(api_url="https://eu.api.smith.langchain.com", api_key=LangChain_API)
    dataset_name = "Lecture Notes Evaluation"
    dataset = client.read_dataset(
        dataset_name=dataset_name
    )
    custom_criteria = {
    # "accuracy": (
    #     "Accuracy: Is the information in the post factually correct "
    #     "and free from errors or misleading statements?"
    # ),
    "relevance": (
        "Relevance: Does the answer accurately and relevantly address the user's prompt (input)?"
        "Please return only a score between 1 and 5, where 1 means 'not relevant at all' and 5 means 'highly relevant'. No text, only score!!!"
    ),
    "clarity": (
        "Clarity & Professionalism: Is the answer clear, concise, well-written, and suitable for lecture notes?"
        "Please return only a score between 1 and 5, where 1 means 'not clear at all' and 5 means 'absolutely clear'. No text, only score!!!"
    )
    }
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw_criteria_evaluator = load_evaluator(
        "criteria", 
        llm=eval_llm,
        criteria=custom_criteria,
        return_only_score=True
    )

    def safe_criteria_evaluator(run, example, **kwargs):
        """A robust wrapper for CriteriaEvalChain output."""
        # Extract strings
        input_str = run.inputs.get("input") or run.inputs.get("question") or str(run.inputs)
        output_str = run.outputs if isinstance(run.outputs, str) else str(run.outputs)

        try:
            # The new interface uses evaluate_strings()
            result = raw_criteria_evaluator.evaluate_strings(
                input=input_str,
                prediction=output_str,
            )
            print(input_str)
            print(result)
        except Exception as e:
            return {
                "key": "criteria_evaluation_error",
                "score": 0.0,
                "comment": f"Evaluation failed: {e}"
            }

        # If the result isn't structured, wrap it
        if isinstance(result, dict) and "reasoning" in result and "key" not in result:
            return {
                "key": "criteria_evaluation_unstructured",
                "score": 0.0,
                "comment": result.get("reasoning", "Unstructured result"),
            }

        return result

    def run_graph_for_eval(example):
        # Construct a full minimal input matching the schema
        graph_input = {
            "input": example["input"],
            "context": [],                  # empty list for context
            "answer_basic": "",             # required but not used yet
            "answer_middle": "",
            "answer_final": "",
            "post_status": ""
        }
        result = chatbot.invoke(graph_input)
        return {"input": example["input"], "output": result["answer_final"]}
    
    return evaluate(
        run_graph_for_eval,
        data=dataset,
        evaluators=[safe_criteria_evaluator],
        experiment_prefix="graph_evaluation_llm_based",
        metadata={"description": "LLM-based evaluation of the lecture notes generator"},
    )

if __name__ == "__main__":
    evaluate_chatbot()
