'''
Description: 
Version: 1.0
Autor: ZhuYichen
Date: 2023-07-03 10:56:48
LastEditors: ZhuYichen
LastEditTime: 2023-07-09 19:22:10
'''
import time

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import gradio as gr
import openai
import ast


class ConversationBot:
    def __init__(self):
        self.system_prompt = None
        self.openai_api_key = None

    def run_text(self, question, llm, change_prompt=None, t=0):
        openai.api_key = self.openai_api_key
        system_prompt = change_prompt if change_prompt else self.system_prompt
        try:
            response = openai.ChatCompletion.create(
                model=llm,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=t,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # res['output'] = res['output'].replace("\\", "/")
            # response = res['output']
            # print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
            #       f"Current Memory: {self.agent.memory.buffer}")
            answer = response.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(60)  # Wait for 1 minute before retrying
            return self.run_text(question, llm, change_prompt)
        print(f"\nQuestion: {question}\nAnswer: {answer}\n")
        return answer

    def init_agent(self, openai_api_key, features):
        self.openai_api_key = openai_api_key
        prompt_dict = dict()
        for feature_type, feature_content in features.items():
            prompt_dict[feature_type] = ''
            for item in feature_content:
                if feature_type == 'subtitle':
                    prompt_dict[feature_type] += 'Second{}: Speaker{}: {}\n'.format(item['begin'], item['speaker'],
                                                                                    item['text'])
                elif feature_type == 'dense':
                    if int(item['begin']) % 5 != 0:
                        continue
                    prompt_dict[feature_type] += 'Second{}: {}\n'.format(int(item['begin'])+1, item['text'])
                elif feature_type == 'frame':
                    prompt_dict[feature_type] += 'Second{}: {}'.format(int(item['begin'])+1, item['text'])
                    # for ocr in features['dense_with_ocr']:
                    #     if ocr['begin'] == item['begin']:
                    #         prompt_dict[feature_type] += ', {}'.format(ocr['text'])
                    prompt_dict[feature_type] += '\n'
        system_prompt = \
            """
                You are a chatbot that conducts conversations based on video descriptions. You mainly answer based on the given description, and you can also modify the content according to the tag information, and you can also answer the relevant knowledge of the person or object contained in the video. The second description is a description for one second, so that you can convert it into time. When describing, please mainly refer to the sceond description. Dense caption is to give content every five seconds, you can disambiguate them in timing. But you don't create a video plot out of nothing.

                Begin!

                The second description of the video is: {}

                The dense caption of the video is: {}

                The subtitle of the video is: {}
            """.format(prompt_dict['frame'], prompt_dict['dense'], prompt_dict['subtitle'], )
        if not openai_api_key.startswith('sk-'):
            print('OPEN_API_KEY ERROR')
        self.system_prompt = system_prompt
        return self.system_prompt

    def init_agent_with_summary(self, openai_api_key, output):
        self.openai_api_key = openai_api_key
        summary = ''
        for item in output['summary']:
            summary += 'Second{} to Second{}: {}\n'.format(item['begin'], item['end'], item['text'])
        subtitle = ''
        for item in output['subtitle']:
            subtitle += 'Second{} to Second{}: Speaker{} said: {}\n'.format(item['begin'], item['end'], item['speaker'], item['text'])
        system_prompt = \
            f"""
                下面是一个视频的若干片段的文字描述以及字幕，描述和字幕都注明了开始时间和结束时间。你需要根据这些信息来回答用户的问题，对于用户的问题，你必须根据信息猜测一个答案。你可以按照以下流程思考：
                1. 找出用户提问的关键词
                2. 在描述中搜索和关键词相关的信息
                3. 根据片段中的内容猜测答案
                下面是文字描述：
                1.描述:
                {summary}
                2.字幕:
                {subtitle}
            """
        if not openai_api_key.startswith('sk-'):
            print('OPEN_API_KEY ERROR')
        self.system_prompt = system_prompt
        print(self.system_prompt)
        return self.system_prompt

    def init_agent_shot(self, openai_api_key, features):
        self.openai_api_key = openai_api_key
        prompt_dict = dict()
        for feature_type, feature_content in features.items():
            prompt_dict[feature_type] = ''
            for item in feature_content:
                if feature_type == 'subtitle':
                    prompt_dict[feature_type] += 'Second{}: Speaker{}: {}\n'.format(item['begin'], item['speaker'],
                                                                                    item['text'])
                elif feature_type == 'dense' or feature_type == 'ocr':
                    continue
                elif feature_type == 'frame':
                    prompt_dict[feature_type] += 'Second{}: {}'.format(item['begin'], item['text'])
                    # for ocr in features['dense_with_ocr']:
                    #     if ocr['begin'] == item['begin']:
                    #         prompt_dict[feature_type] += ', {}'.format(ocr['text'])
                    prompt_dict[feature_type] += '\n'

        system_prompt = \
            """
                你需要根据用户提供的视频片段的描述来总结视频，描述分为三部分
                1.帧描述：帧描述提供了每一秒视频中的主体和事件
                2.字幕：字幕提供了某个时间点，视频中的人说的话，你可以根据它来推断视频中发生了什么。
                你需要注意视频片段中出现的对话和行为。下面是一个参考案例：
                '''
                用户提供的描述：
                1. 
                帧描述:
                Second1: a man walking down a street next to parked cars
                Second2: a man riding a scooter on a street with cars and pedestrians
                Second3: a busy street with people riding mopeds, motorcycles and cars
                Second4: a man riding a scooter down a busy street with cars and people on motorcycles
                Second5: a man riding a black moped down a busy street with cars
                Second6: a man riding a scooter down a street next to cars
                Second7: a man riding a black moped down a street next to cars on the sidewalk
                Second8: a man riding a moped down a busy street with parked cars on the sidewalk
                Second9: a man riding a motor scooter down a street next to a sidewalk
                Second10: a man riding a moped down a street with watermelons
                2.
                字幕：
                Second0: Speaker1: 有一个人前来买瓜
                '''
                参考答案：
                一个男人骑着摩托车沿着街道行驶，周围停着一些汽车。街道上有许多人骑着摩托车、机动车和自行车行驶。然后这个骑着黑色摩托车的男人到了一个西瓜摊边上。
                '''
                下面你将收到用户提供的描述：
            """
        question = \
            """
                1. 
                帧描述:
                {}
                2.
                字幕:
                {}
                你的总结：
            """.format(prompt_dict['frame'], prompt_dict['subtitle'])
        if not openai_api_key.startswith('sk-'):
            print('OPEN_API_KEY ERROR')
        self.system_prompt = system_prompt
        return self.system_prompt, question

    def evaluate_qa(self, openai_api_key, qa, llm):
        openai.api_key = openai_api_key
        system_prompt = \
            """
                对比标准答案，评估以下视频问答模型的预测准确性，按照0-5的评分标准：

                ‘’’
                
                0：预测答案与标准答案完全不一致或无关。
                
                1：预测答案与标准答案部分一致，但主要的信息都没有涵盖。
                
                2：预测答案中包含了部分标准答案，但并没有完全回答问题。
                
                3：预测答案包含了标准答案的大部分内容，但缺少一些关键信息。
                
                4：预测答案和标准答案几乎完全一致，只是在细节上有一些小的遗漏。
                
                5：预测答案和标准答案完全一致。
                
                ‘’’
                你需要给出评分的理由，再进行评分。
                
                你的回答必须以字典格式写出：
                
                {'reason': 评分理由,'score': 分数,}
                
                问题、标准答案以及预测答案分别如下：
            """
        response = openai.ChatCompletion.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": qa
                }
            ],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].message.content
        print(f"\nAnswer: {answer}\n")
        answer = answer.strip("{}").replace("'", "")
        try:
            score_part = answer.split(", ")[-1]
            reason_part = answer[:-len(score_part)].rstrip(", ")
            dict_answer = dict(item.split(": ") for item in [reason_part, score_part])
            # dict_answer = dict(item.split(": ") for item in answer.split(", "))
        except Exception as e:
            dict_answer = {
                'reason': '返回评分格式错误',
                'score': 0,
            }
            print(e)
        return dict_answer


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
