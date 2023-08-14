from collections import defaultdict

import math
import os
import json

from models.tag2text import tag2text_caption
from util import *
from chatbot import *
from projects.videochat.models.load_internvideo import *
from simplet5 import SimpleT5
from models.grit_model import DenseCaptioning
# from paddleocr import PaddleOCR
import whisper
from models.transnetv2 import Shot, ShotProcessor
from datetime import datetime
import configparser
from projects.videochat.models.subtitle import RequestApi, ProcessSubtitle

config = configparser.ConfigParser()
config.read('configs.ini')
args = {
    'videos_path': config.get('Arguments', 'videos_path'),
    'openai_api_key': os.environ["OPENAI_API_KEY"],
    'output_path': config.get('Arguments', 'output_path'),
    'images_path': config.get('Arguments', 'images_path'),
    'evaluate_path': config.get('Arguments', 'evaluate_path'),
    'appid': config.get('Arguments', 'appid'),
    'secret_key': config.get('Arguments', 'secret_key'),
    'segment_length': int(config.get('Arguments', 'segment_length')),
    'remarks': config.get('Arguments', 'remarks'),
    'llm': config.get('Arguments', 'llm'),
    'predict': config.get('Arguments', 'predict') == 'True',
    'evaluate': config.get('Arguments', 'evaluate') == 'True',
    'mode': config.get('Arguments', 'mode'),
    'qa_mode': config.get('Arguments', 'qa_mode'),
}
os.environ["CUDA_VISIBLE_DEVICES"] = config.get('Arguments', 'device')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VideoChat:
    def __init__(self):
        self.bot = ConversationBot()

    def load_model(self):
        image_size = 384
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])

        # define model
        self.shot_model = Shot("pretrained_models/transnetv2-pytorch-weights.pth")

        self.model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size,
                                      vit='swin_b')
        self.model.eval()
        self.model = self.model.to(device)
        print("[INFO] initialize caption model success!")

        self.model_T5 = SimpleT5()
        if torch.cuda.is_available():
            self.model_T5.load_model(
                "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=True)
        else:
            self.model_T5.load_model(
                "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=False)
        print("[INFO] initialize summarize model success!")
        # action recognition
        self.intern_action = load_intern_action(device, pretrained="pretrained_models/uniformerv2/k400+k710_uniformerv2_b16_8x224.pyth")
        self.trans_action = transform_action()
        self.topil = T.ToPILImage()
        print("[INFO] initialize InternVideo model success!")

        self.dense_caption_model = DenseCaptioning(device)
        self.dense_caption_model.initialize_model()
        print("[INFO] initialize dense caption model success!")

        self.whisper_model = whisper.load_model("large")
        print("[INFO] initialize whisper model success!")

    def inference_second(self, video_path, input_tag):
        # shot
        shot_result = self.shot_model.inference(video_path)
        # Whisper
        whisper = list()
        try:
            whisper_result = self.whisper_model.transcribe(video_path)
            for segment in whisper_result['segments']:
                whisper.append({
                    'begin': int(segment['start']),
                    'end': int(segment['end']),
                    'text': segment['text'],
                })
        except Exception as e:
            print(e)

        # 讯飞API
        subtitle = list()
        try:
            api = RequestApi(appid=args['appid'],
                             secret_key=args['secret_key'],
                             upload_file_path=video_path)

            api.get_result()
            subtitle = api.result2text(return_list=True)
        except Exception as e:
            print(e)

        data = loadvideo_decord_origin(video_path)
        tmp = []
        for i, img in enumerate(data):
            tmp.append(self.transform(img).to(device).unsqueeze(0))
        # dense caption
        dense_caption = list()
        dense_caption_with_pos = list()
        dense_foot = 1
        dense_index = np.arange(0, len(data), dense_foot)
        original_images = data[dense_index, :, :, ::-1]
        with torch.no_grad():
            for index, original_image in zip(dense_index, original_images):
                new_caption_only_name, new_caption = self.dense_caption_model.run_caption_tensor(original_image,
                                                                                                 video_path, index,
                                                                                                 args['images_path'])
                dense_caption.append({
                    'begin': index,
                    'text': new_caption_only_name,
                })

                dense_caption_with_pos.append({
                    'begin': index,
                    'text': new_caption,
                })
        del data, original_images
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Frame Caption
        image = torch.cat(tmp).to(device)

        self.model.threshold = 0.68
        if input_tag == '' or input_tag == 'none' or input_tag == 'None':
            input_tag_list = None
        else:
            input_tag_list = [input_tag.replace(',', ' | ')]
        with torch.no_grad():
            caption, tag_predict = self.model.generate_sublists(image, tag_input=input_tag_list, max_length=50,
                                                                return_tag_predict=True)
            frame_caption = list()
            for i, j in enumerate(caption):
                frame_caption.append({
                    'begin': i,
                    'text': j,
                })
            tag_1 = set(tag_predict)
            synth_caption = self.model_T5.predict('. '.join(caption))

        del image, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        features = dict()
        features['shot'] = shot_result.tolist()
        features['subtitle'] = subtitle
        features['whisper'] = whisper
        features['dense'] = dense_caption
        features['dense_with_pos'] = dense_caption_with_pos
        features['frame'] = frame_caption
        features['synth_caption'] = synth_caption
        features['tag'] = list(tag_1)

        return features


class InputVideo:
    def __init__(self, videos_path):
        self.output_path = None
        self.cur_time = None
        self.evaluate_path = ''
        self.videos_path = videos_path
        self.video_chat = VideoChat()
        self.videos = list()
        self.questions = list()
        self.features = list()
        self.exist_features = True
        print('Test video numbers:', len(os.listdir(self.videos_path)))
        for item in os.listdir(self.videos_path):
            # if int(item) != 0:
            #     continue
            json_path = os.path.join(self.videos_path, item, 'data.json')
            video_path = os.path.join(self.videos_path, item, 'video.mp4')
            features_path = os.path.join(self.videos_path, item, 'features_test.json')
            if not os.path.exists(features_path):
                self.exist_features = False
            self.features.append(features_path)
            self.questions.append(json_path)
            self.videos.append(video_path)

        self.extract_features()

    def extract_features(self):
        if not self.exist_features:
            self.video_chat.load_model()
        for index, video_path in enumerate(self.videos):
            if not os.path.exists(self.features[index]):
                try:
                    features = self.video_chat.inference_second(video_path, '')
                    # Merge Whisper and Xunfei
                    process_subtitle = ProcessSubtitle(features)
                    features['merged_subtitle'] = process_subtitle.merge_whisper_and_xunfei()
                    # Shot
                    shot_processor = ShotProcessor()
                    features['time_intervals'] = shot_processor.shot(video_path, features['shot'])
                    with open(self.features[index], 'w') as file:
                        json.dump(features, file, indent=4, ensure_ascii=False, cls=CustomEncoder)
                except Exception as e:
                    print(e)
                    continue

    def start_test(self):
        self.cur_time = datetime.now()
        self.output_path = os.path.join(args['output_path'], self.cur_time.strftime("%Y%m%d%H%M%S"))
        for index, video_path in enumerate(self.videos):
            if os.path.exists(self.features[index]):
                with open(self.features[index], 'r') as file:
                    data = json.load(file)
                    features = data
            else:
                print('features.json not found')
                continue
            if args['mode'] == 'normal':
                self.video_chat.bot.init_agent(args['openai_api_key'], features)
                self.qa_test(index, features, video_path)
            elif args['mode'] == 'shot':
                begin_time = 0
                summary_list = list()
                for shot_time in features['time_intervals']:
                    end_time = shot_time
                    if end_time == 0:
                        continue
                    shot_features = dict()
                    for feature_type, feature_content in features.items():
                        if len(feature_content) == 0 or not isinstance(feature_content[0], dict) or 'begin' not in feature_content[0]:
                            continue
                        shot_features[feature_type] = list()
                        for item in feature_content:
                            if begin_time <= item['begin'] < end_time:
                                shot_features[feature_type].append(item)
                    if len(shot_features['frame']) == 0:
                        continue
                    # dense_with_ocr = find_text_in_dense(shot_features)
                    # shot_features['dense_with_ocr'] = dense_with_ocr
                    prompt, question = self.video_chat.bot.init_agent_shot(args['openai_api_key'],
                                                                           shot_features)
                    summary = self.video_chat.bot.run_text(question, args['llm'], None, t=1)
                    summary_list.append({
                        'begin': begin_time,
                        'end': end_time,
                        'text': summary,
                    })
                    begin_time = shot_time
                output = dict()
                output['summary'] = summary_list
                output['subtitle'] = features['merged_subtitle']
                if self.output_path:
                    folder = os.path.basename(os.path.dirname(video_path))
                    folder_path = os.path.join(self.output_path, folder)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_path = os.path.join(folder_path, 'summary.json')
                    with open(save_path, "w") as json_file:
                        json.dump(summary_list, json_file, indent=4, ensure_ascii=False)
                prompt = self.video_chat.bot.init_agent_with_summary(args['openai_api_key'], output)
                self.qa_test(index, prompt, video_path)

    def qa_test(self, index, features, video_path):
        output = dict()
        qa_output = list()

        with open(self.questions[index]) as file:
            data = json.load(file)
            output['video_name'] = data['video_name']
            output['test_time'] = self.cur_time.strftime("%Y-%m-%d %H:%M:%S")
            output['remarks'] = args['remarks']
            for qa in data['qa']:
                if args['qa_mode'] == 'think':
                    question = '你可以按照以下流程思考：1. 找出用户提问的关键词;2. 在描述中搜索和关键词相关的描述;3. 根据片段中的内容推理答案。\n请你给出你的思考过程：推理' + \
                               qa['q']
                else:
                    question = qa['q']
                answer = qa['a']
                infer_answer = self.video_chat.bot.run_text(question, args['llm'])
                if args['qa_mode'] == 'think':
                    question = {
                        'q': qa['q'],
                        'predict': infer_answer
                    }
                    change_prompt = \
                        '''
                            你需要对指定的回答进行总结。
                            用户将输入以下格式的内容：
                            {
                                "q": "华强用刀做了哪些事情？",
                                "predict": "根据描述和字幕中的信息，我们得知视频中出现了一个男人使用刀的场景。以下是我根据描述和字幕提供的信息的猜测：\n\n1. 在Second74 to Second76这段视频中，一个男人在一个市场的水果摊前切割了一个西瓜。可以推测，这个男人使用刀来切开了西瓜。\n\n所以，根据视频的描述，我们可以推测华强使用刀来切割了一个西瓜。"
                            },
                            其中q是问题，predict是一个包含了思考过程的回答。你需要对这个回答进行总结，去掉其中的思考过程，保留主要的答案。
                            对于上面这个例子，你的总结应该是：“华强使用刀来切割了一个西瓜。”

                            下面是用户的输入：
                        '''
                    infer_answer = self.video_chat.bot.run_text(str(question), args['llm'], change_prompt)
                if not infer_answer:
                    print('token过长，无法回答')
                    continue
                qa_output.append({'q': qa['q'], 'a': answer, 'predict': infer_answer})
            output['qa'] = qa_output
            output['features'] = features
        if self.output_path:
            folder = os.path.basename(os.path.dirname(video_path))
            folder_path = os.path.join(self.output_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path, 'output.json')
            with open(save_path, "w") as json_file:
                json.dump(output, json_file, indent=4, ensure_ascii=False, cls=CustomEncoder)

    def evaluate_by_chatgpt(self):
        if args['predict'] and args['evaluate']:
            all_predict_results = os.listdir(args['output_path'])
            int_list = [int(s) for s in all_predict_results]
            self.evaluate_path = os.path.join(args['output_path'], str(max(int_list)))
        elif not args['predict'] and args['evaluate']:
            self.evaluate_path = args['evaluate_path']

        evaluate_result = dict()
        total_score = 0
        qa_number = 0
        evaluate_save_path = os.path.join(self.evaluate_path, 'evaluate_result.json')
        answer_list = []
        for output_folder in os.listdir(self.evaluate_path):
            # if int(output_folder) != 14:
            #     continue
            if output_folder.endswith(".json"):
                continue
            output_file = os.path.join(self.evaluate_path, output_folder, 'output.json')
            if not os.path.exists(output_file):
                continue
            with open(output_file) as file:
                data = json.load(file)
                cur_score = 0
                cur_number = 0
                cur_result = dict()
                cur_answer = list()
                for qa in data['qa']:
                    try:
                        answer = self.video_chat.bot.evaluate_qa(args['openai_api_key'], str(qa), args['llm'])
                        total_score += int(answer['score'])
                        qa_number += 1
                        cur_score += int(answer['score'])
                        cur_number += 1
                        cur_answer.append(answer)
                    except Exception as e:
                        print(e)
                cur_result['video'] = output_folder
                cur_result['mean_score'] = cur_score / cur_number
                cur_result['answer'] = cur_answer
                answer_list.append(cur_result)
        evaluate_result['mean_score'] = total_score / qa_number
        evaluate_result['remarks'] = args['remarks']
        evaluate_result['answer_list'] = answer_list
        with open(evaluate_save_path, "w") as json_file:
            json.dump(evaluate_result, json_file, indent=4, ensure_ascii=False)


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.int64):
            return int(o)  # Convert int64 to Python int
        return super().default(o)


def main():
    input_video = InputVideo(args['videos_path'])
    if args['predict']:
        input_video.start_test()
    if args['evaluate']:
        input_video.evaluate_by_chatgpt()


if __name__ == "__main__":
    main()
