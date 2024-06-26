# from datasets.base import BaseTemplate

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

load_dotenv()

_examples = [
    {
        "input": "/t".join(
            [
                "In which year was the seminal Human Development Report published?",
                "It was published in 1990.",
            ]
        ),
        "output": "/t".join(
            [
                "중요한 인간 개발 보고서(Human Development Report)는 몇 년도에 발행되었나요?",
                "보고서는 1990년에 발행되었습니다.",
            ]
        ),
    },
    {
        "input": "/t".join(
            [
                "Sam wants to go to bed.",
                "Tesla makes the coolest car in the world.",
            ]
        ),
        "output": "/t".join(
            [
                "민호는 자려고 합니다.",
                "기아는 세상에서 가장 멋진 차를 만듭니다.",
            ]
        ),
    },
]
_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=_examples,
    example_prompt=_example_prompt,
)

_instructions = """당신은 한국어 번역가로서 영어 문장을 한국어로 번역하고, 윤문해야 합니다. 다음 <guidelines>을 지켜 번역하세요.

<guidelines>
1. 문맥을 반영하여 번역하세요. 여기서 문맥이란 <input> 전체의 문맥을 의미합니다. 예를 들어 'organism'은 문맥에 따라 '유기체, 생명체, 생명, 유기적 조직체' 등 다양하게 번역될 수 있습니다.
2. 번역문은 한국어 원어민이 쉽게 이해할 수 있도록 한국인 입장에서 자연스러운 표현으로 이뤄져야 합니다. 영어 원문의 의미를 정확하게 전달하되, 직역이 어색할 경우 자연스러운 표현으로 번역하세요.
3. 10대 청소년도 이해할 수 있을 정도로 문장을 윤문하세요. 영어 원문의 톤은 유지해야 합니다.
4. 문화적 차이를 고려하여 한국어 표현을 선택하세요. 예를 들어 'kick the bucket'은 '세상을 떠나다'로 번역할 수 있습니다.
5. 사람 이름은 한국식 이름으로 번역합니다.
6. 기업 이름은 유사 업종의 가장 유명한 한국 기업으로 변경합니다. 
7. 한국어의 격식체를 사용합니다. 격식체란 다음과 같이 '~있습니다', '~니다', '~할까요?' 등의 문장 끝말을 사용하는 것을 말합니다.
8. 전문 용어나 어려운 용어는 영어 원문 단어를 중괄호 안에 넣어 번역하세요. 예를 들어 '항정신성 약물(Antisychotics)은 ...'와 같이 표기합니다.
</guidelines>

<examples>
input: In which year was the seminal Human Development Report published?/tIt was published in 1990.
output: 중요한 인간 개발 보고서(Human Development Report)는 몇 년도에 발행되었나요?/t보고서는 1990년에 발행되었습니다.
---
input: Sam wants to go to bed./tTesla makes the coolest car in the world.
output: 민호는 자려고 합니다./t기아는 세상에서 가장 멋진 차를 만듭니다.
</examples>"""

final_prompt = ChatPromptTemplate.from_messages(
    [("system", _instructions), _few_shot_prompt, ("human", "<input>{input}</input>")]
)

# final_prompt = ChatPromptTemplate.from_template(_instructions)
# llm = ChatAnthropic(
#     model="claude-3-5-sonnet-20240620",  # claude-3-opus-20240220
#     temperature=0.1,
# )
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

chain = final_prompt | llm
output = chain.invoke(
    {
        "input": "/t".join(
            [
                "The quick brown fox jumps over the lazy dog.",
                "The passage was coined by Noam Chomsky.",
            ]
        ),
    }
)

print(output)
