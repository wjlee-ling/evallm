from datasets.utils import concatenate_columns, load_from_path
import anthropic
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
client = anthropic.Anthropic()


def _complete_anthropic(prompt: str):
    _prompt = [
        # {"role": "system", "content": "Follow the following instructions"},
        {"role": "user", "content": prompt},
    ]
    messages = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=_prompt,
    )
    return messages


def _format_examples(examples: list[dict]):
    return "\n\n".join(
        [f"user: {ex['input']}\nassistant: {ex['output']}" for ex in examples]
    )


def _parse_response(response, columns: list) -> list:
    parsed = response.content[0].text.split(DELIMITER)
    new_row = {col: resp for col, resp in zip(columns, parsed)}
    return new_row


def translate_df(df, columns, prompt) -> pd.DataFrame:
    new_rows = []
    if columns is None:
        # If columns are not specified, use all columns
        columns = list(range(len(df.columns)))

    for idx, row in tqdm(df.iterrows()):
        row_as_string = concatenate_columns(row, columns)
        final_prompt = str(prompt.format(input=row_as_string)).lstrip("Human: ")
        response = _complete_anthropic(final_prompt)
        parsed = _parse_response(response, columns)
        new_rows.append(parsed)

    new_df = pd.DataFrame(new_rows, columns=columns)
    return new_df


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
    # _example_prompt = ChatPromptTemplate.from_messages(
    #     [("user", "{input}"), ("assistant", "{output}")]  # 🚨 Anthropic: user/assistant
    # )

    # _few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     examples=_examples,
    #     example_prompt=_example_prompt,
    # )

    _template = """당신은 한국어 번역가로서 영어 문장을 한국어로 번역하고 윤문해야 합니다. 다음 <guidelines>을 지켜 번역하세요.

    <guidelines>
    1. 문맥을 반영하여 번역하세요. 여기서 문맥이란 <input> 전체의 문맥을 의미합니다. 예를 들어 'organism'은 문맥에 따라 '유기체, 생명체, 생명, 유기적 조직체' 등 다양하게 번역될 수 있습니다.
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
    </examples>

    <input>
    {input}
    </input>"""

    _few_shot_prompt = _format_examples(_examples)
    template = ChatPromptTemplate.from_template(_template)
    prompt = template.partial(examples=_few_shot_prompt)

    for p in tqdm(load_from_path(path)):
        if p.suffix == ".csv":
            df = pd.read_csv(p, header=None if headless else 0)
            if headless:
                df.columns = [str(col) for col in df.columns]
            new_df = translate_df(df, columns, prompt)

            # fill in missing columns
            new_df = new_df.combine_first(df)
            new_df.to_csv(p.with_suffix(".translated.csv"), index=False)

        elif p.suffix == ".xlsx":
            df = pd.read_excel(p, header=None if headless else 0)
            if headless:
                df.columns = [str(col) for col in df.columns]
            new_df = translate_df(df, columns, prompt)
            new_df = new_df.combine_first(df)
            new_df.to_csv(p.with_suffix(".translated.csv"), index=False)
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
