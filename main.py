from Job_parser.core.parser import IntelligentResumeParser
from Job_parser.core.knowledge_graph import OllamaGraph


def extract_main_summaries(results):
    return [result.get("summary", "").strip() for result in results if result.get("summary", "").strip()]
if __name__ == "__main__":
    skills = ["deep learning","data mining","Python ","SAS","machine learning ","TensorFlow","Keras","Scikit-learn"
              ,"NLP ","PowerBI","SQL","AWS"]

    paragraph = """ 
• Partner with cross-functional teams to design and implement data-driven solutions that align with business goals  
• Gather, clean, and integrate data from disparate sources to support analytical workflows  
• Utilize predictive modeling techniques to identify trends and inform strategic planning  
• Act as an internal consultant to enhance data fluency across non-technical departments  
• Establish secure data pipelines and manage cloud infrastructure access  
• Evaluate emerging technologies and incorporate best practices into existing analytics frameworks  
• Lead technical initiatives and coach junior analysts in project execution

"""

    parser = IntelligentResumeParser(skills=skills)
    results = parser.analyze_responsibilities(paragraph)

    print("\n=== Precise Responsibility Analysis ===")
    main_summaries = []

    for i, result in enumerate(results, 1):
        summary = result.get("summary", "Responsibility summary")
        main_summaries.append(summary)

        print(f"{i}. {summary}")
        tools = result.get("tools")
        confidences = result.get("tool_confidence")

        if tools and confidences:
            print("   Tools used:")
            for tool, confidence in zip(tools, confidences):
                print(f"     - {tool} ({confidence}%)")
        else:
            print("   Tools used: None")

        subtasks = result.get("subtasks", [])
        if subtasks:
            print("   Subtasks:")
            for j, sub in enumerate(subtasks, 1):
                print(f"      {j}. {sub.get('task', 'Task')}")
                expl = sub.get("explanation", "")
                if expl:
                    print(f"          - {expl}")
        print()

    print("\n-- Running LLM-Based Knowledge Graph Analysis --")
    graph = OllamaGraph("knowledge_graph.json")
    graph.load()


    graph.process_main_actions(main_summaries)


    graph.deduplicate_nodes()


    graph.save()

    print("-- Graph Analysis Completed --")


