import json

import lotus
from lotus.types import (
    SemanticExtractPostprocessOutput,
    SemanticFilterPostprocessOutput,
    SemanticMapPostprocessOutput,
)


def cot_postprocessor(llm_answers: list[str]):
    outputs: list[str | None] = []
    explanations: list[str | None] = []
    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]

        explanations.append(reasoning)
        outputs.append(answer)

    return outputs, explanations


def map_postprocess_cot(llm_answers: list[str]) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator with CoT reasoning.

    Args:
        llm_answers (list[str]): The list of llm answers.

    Returns:
        SemanticMapPostprocessOutput
    """
    outputs: list[str] = []
    explanations: list[str | None] = []

    for llm_answer in llm_answers:
        reasoning_idx = llm_answer.find("Reasoning:\n")
        if reasoning_idx == -1:
            reasoning_idx = 0
        else:
            reasoning_idx += len("Reasoning:\n")

        answer_idx = llm_answer.find("Answer:")
        reasoning = llm_answer[reasoning_idx:answer_idx].rstrip("\n").lstrip("\n")
        answer = llm_answer[answer_idx + len("Answer:") :]
        outputs.append(answer)
        explanations.append(reasoning)

    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)


def map_postprocess(llm_answers: list[str], cot_reasoning: bool = False) -> SemanticMapPostprocessOutput:
    """
    Postprocess the output of the map operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticMapPostprocessOutput
    """
    if cot_reasoning:
        return map_postprocess_cot(llm_answers)

    outputs: list[str] = llm_answers
    explanations: list[str | None] = [None] * len(llm_answers)
    return SemanticMapPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)


def extract_postprocess(llm_answers: list[str]) -> SemanticExtractPostprocessOutput:
    """
    Postprocess the output of the extract operator to extract the schema.

    Args:
        llm_answers (list[str]): The list of llm answers containing the extract.

    Returns:
        SemanticExtractPostprocessOutput
    """
    extract_data = []
    for llm_answer in llm_answers:
        try:
            output = json.loads(llm_answer)
        except json.JSONDecodeError:
            lotus.logger.info(f"\t Failed to parse: {llm_answer}")
            output = {}

        output = {key: str(value) for key, value in output.items()}
        extract_data.append(output)

    return SemanticExtractPostprocessOutput(raw_outputs=llm_answers, outputs=extract_data)


def filter_postprocess(
    llm_answers: list[str],
    default: bool = True,
) -> SemanticFilterPostprocessOutput:
    """
    Postprocess the output of the filter operator.

    Args:
        llm_answers (list[str]): The list of llm answers.
        default (bool): The default value to use if we fail to parse the answer.
        cot_reasoning (bool): Whether there is CoT reasoning.

    Returns:
        SemanticFilterPostprocessOutput
    """
    outputs, explanations = cot_postprocessor(llm_answers)

    def process_outputs(answer):
        if answer is None:
            lotus.logger.info(f"\t Failed to parse {answer}: defaulting to {default}")
            return default

        if "True" in answer:
            return True
        elif "False" in answer:
            return False
        else:
            lotus.logger.info(f"\t Failed to parse {answer}: defaulting to {default}")
            return default

    outputs = [process_outputs(answer) for answer in outputs]

    return SemanticFilterPostprocessOutput(raw_outputs=llm_answers, outputs=outputs, explanations=explanations)
