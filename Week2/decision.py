def recommend_approach(scenario: dict) -> str:
    data_changes = scenario["data_changes_frequently"]
    specific_format = scenario["need_specific_output_format"]
    budget = scenario["budget"]
    latency = scenario["latency_sensitive"]
    knowledge = scenario["knowledge_type"]

    if knowledge == "both":
        if budget in ["medium", "high"]:
            return "RAG + Fine-Tuning: You need dynamic facts retrieval plus customized behavior/output style."
        else:
            return "RAG: Both would help, but budget is low, so prioritize retrieval over training."

    elif knowledge == "factual":
        if data_changes:
            return "RAG: Knowledge changes frequently, so external retrieval is better than retraining."
        elif latency and budget == "low":
            return "Prompt Engineering only: Facts are stable and constraints are simple, so prompting may be enough."
        else:
            return "RAG: External knowledge retrieval improves factual accuracy."

    elif knowledge == "behavioral":
        if specific_format:
            if budget == "high":
                return "Fine-Tuning: Strong formatting/style consistency benefits from model training."
            else:
                return "Prompt Engineering only: Formatting can be controlled through carefully designed prompts."
        else:
            return "Prompt Engineering only: Behavior tweaks alone often don't require fine-tuning."

    return "Prompt Engineering only: Default lightweight solution."

if __name__ == "__main__":

    scenarios = [
        {
            "name": "Customer support chatbot with daily policy updates",
            "data_changes_frequently": True,
            "need_specific_output_format": False,
            "budget": "medium",
            "latency_sensitive": False,
            "knowledge_type": "factual"
        },

        {
            "name": "Legal document formatter",
            "data_changes_frequently": False,
            "need_specific_output_format": True,
            "budget": "high",
            "latency_sensitive": True,
            "knowledge_type": "behavioral"
        },

        {
            "name": "Personal tutor with company-specific documents and teaching style",
            "data_changes_frequently": True,
            "need_specific_output_format": True,
            "budget": "high",
            "latency_sensitive": False,
            "knowledge_type": "both"
        },

        {
            "name": "Simple email rewriter",
            "data_changes_frequently": False,
            "need_specific_output_format": True,
            "budget": "low",
            "latency_sensitive": True,
            "knowledge_type": "behavioral"
        },

        {
            "name": "Static FAQ bot",
            "data_changes_frequently": False,
            "need_specific_output_format": False,
            "budget": "low",
            "latency_sensitive": True,
            "knowledge_type": "factual"
        }
    ]

    for s in scenarios:
        print(f"{s['name']} -> {recommend_approach(s)}")
        print(" ")