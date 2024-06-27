from datasets.utils import concatenate_columns, load_from_path
import anthropic
import json
import re
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from tqdm import tqdm

load_dotenv()

DELIMITER = "\t"
CHUNK_SIZE = 3
client = anthropic.Anthropic()


def _complete_anthropic(prompt: str, system_message: str = None):
    _prompt = [
        {"role": "user", "content": prompt},
    ]
    messages = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        messages=_prompt,
        system=system_message,
    )
    return messages


def _format_examples(examples: list[dict]):
    input_examples = json.dumps(
        {idx: f"{ex['input']}" for idx, ex in enumerate(examples)}
    )
    output_examples = json.dumps(
        {idx: f"{ex['output']}" for idx, ex in enumerate(examples)}
    )

    return "\n\n".join([f"user: {input_examples}", f"assistant: {output_examples}"])


def _parse_response(response, columns: list) -> dict[int, dict]:
    """Returns
    {
        0: {"column1": "value1", "column2": "value2", "column3": "value3"},
    }
    """
    parsed = {}
    try:
        for idx, row_data in eval(response.content[0].text).items():
            values = row_data.split(DELIMITER)
            parsed[idx] = {col: val for col, val in zip(columns, values)}

    except SyntaxError:
        # can encounter SyntaxError when `eval`ing the response
        values = response.content[0].text.strip(" {}").split(DELIMITER)
        for idx, value in enumerate(values):
            value = re.sub(r"^\"?[0-9]+[\"\s]?:\s?", "", value.strip('"'))
            if idx // CHUNK_SIZE not in parsed:
                parsed[idx // CHUNK_SIZE] = {}
            parsed[idx // CHUNK_SIZE].update(
                {col: value for i, col in enumerate(columns) if i == idx % CHUNK_SIZE}
            )

    return parsed


def _format_chunk(chunk: pd.DataFrame, columns) -> dict[int, str]:
    """
    {
        0: "column1\tcolumn2\tcolumn3",
    }
    """
    chunk_in_format = {}
    for idx, row in chunk.iterrows():
        row_data = concatenate_columns(row, columns)
        chunk_in_format[idx] = row_data
    return json.dumps(chunk_in_format)


def translate_df(df, columns, prompt, path):
    instruct_prompt = prompt.lstrip("Human: ")
    chunks = [df.iloc[i : i + CHUNK_SIZE] for i in range(0, df.shape[0], CHUNK_SIZE)]

    with open(path, "a") as f:
        print(f"Translating {path.name}...")
        if f.tell() == 0:
            f.write(",".join(columns) + "\n")

        for chunk in tqdm(chunks):
            formatted_rows = _format_chunk(chunk, columns)
            response = _complete_anthropic(
                prompt=formatted_rows, system_message=instruct_prompt
            )
            parsed = _parse_response(response, columns)
            pd.DataFrame(parsed).T.to_csv(f, header=False, index=False)


def main(path, columns, headless):

    _examples = [
        {
            "input": DELIMITER.join(
                [
                    "In which year was the seminal Human Development Report published?",
                    "It was published in 1990.",
                ]
            ),
            "output": DELIMITER.join(
                [
                    "중요한 인간 개발 보고서(Human Development Report)는 몇 년도에 발행되었나요?",
                    "보고서는 1990년에 발행되었습니다.",
                ]
            ),
        },
        {
            "input": DELIMITER.join(
                [
                    "Sam wants to go to bed.",
                    "Tesla makes the coolest car in the world.",
                ]
            ),
            "output": DELIMITER.join(
                [
                    "민호는 자려고 합니다.",
                    "기아는 세상에서 가장 멋진 차를 만듭니다.",
                ]
            ),
        },
    ]
    # _example_[pr]ompt = ChatPromptTemplate.from_messages(
    #     [("user", "{input}"), ("assistant", "{output}")]  # 🚨 Anthropic: user/assistant
    # )

    # _few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     examples=_examples,
    #     example_prompt=_example_prompt,
    # )

    _template = """당신은 한국어 번역가로서 영어 문장을 한국어로 번역하고 윤문해야 합니다. 원문은 문장 인덱스별로 질문과 답변 선택지로 구성되어 있습니다.\
    질문과 답변 선택지는 \t으로 분리되어 있으며, 첫 value는 질문이고 나머지 value는 답변 선택지입니다. 번역문은 각 문장을 번역한 후 \t로 구분하여 작성하세요.\
    다음 <guidelines>을 지켜 번역하세요.

    <guidelines>
    1. 문맥을 반영하여 번역하세요. 여기서 문맥이란 <input>내 해당 인덱스의 문장 전체를 의미합니다. 예를 들어 'organism'은 문맥에 따라 '유기체, 생명체, 생명, 유기적 조직체' 등 다양하게 번역될 수 있습니다.
    2. 번역문은 한국어 원어민이 쉽게 이해할 수 있도록 한국인 입장에서 자연스러운 표현으로 이뤄져야 합니다. 영어 원문의 의미를 정확하게 전달하되, 직역이 어색할 경우 자연스러운 표현으로 번역하세요.
    3. 10대 청소년도 이해할 수 있을 정도로 문장을 윤문하세요. 영어 원문의 톤은 유지해야 합니다.
    4. 문화적 차이를 고려하여 한국어 표현을 선택하세요. 예를 들어 'kick the bucket'은 '세상을 떠나다'로 번역할 수 있습니다.
    5. 사람 이름은 한국식 이름으로 번역합니다. 기업 이름은 유사 업종의 가장 유명한 한국 기업으로 변경합니다. 
    6. 한국어의 격식체를 사용합니다. 격식체란 다음과 같이 '~있습니다', '~니다', '~할까요?' 등의 문장 끝말을 사용하는 것을 말합니다.
    7. 전문 용어나 어려운 용어는 영어 원문 단어를 중괄호 안에 넣어 번역하세요. 예를 들어 '항정신성 약물(Antisychotics)은 ...'와 같이 표기합니다.
    8. 문제 형식의 텍스트를 번역할 때는 정답을 추론하거나 표시하지 마세요. 주어진 텍스트만 번역하고 추가적인 설명이나 해석을 덧붙이지 마세요. 번역 시 의심스러운 부분이 있더라도 추측하지 마세요.
    9. 원문의 형식을 그대로 유지하세요. 선택지, 정답 표시 등 원문의 구조적 요소를 변경하지 마세요.
    </guidelines>
    
    <examples>
    {examples}
    </examples>"""

    _few_shot_prompt = _format_examples(_examples)
    instruct_template = ChatPromptTemplate.from_template(_template)
    instruct_prompt = instruct_template.format(examples=_few_shot_prompt)

    for p in tqdm(load_from_path(path)):
        if p.suffix == ".csv":
            df = pd.read_csv(p, header=None if headless else 0)
        elif p.suffix == ".xlsx":
            df = pd.read_excel(p, header=None if headless else 0)
        if headless:
            df.columns = [str(col) for col in df.columns]
        columns = df.columns.tolist()  # use all the columns
        translate_df(df, columns, instruct_prompt, p.with_suffix(".translated.csv"))

        # # fill in missing columns
        # new_df = new_df.combine_first(df)
        # new_df.to_csv(p.with_suffix(".translated.csv"), index=False)
        import time

        time.sleep(60)


# chain = final_prompt | llm
# output = chain.invoke(
#     {
#         "input": "/t".join(
#             [
#                 "The quick brown fox jumps over the lazy dog.",
#                 "The passage was coined by Noam Chomsky.",
#             ]
#         ),
#     }
# )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--columns", nargs="+")
    args = parser.parse_args()
    output = main(path=args.path, columns=args.columns, headless=args.headless)
