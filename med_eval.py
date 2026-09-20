import random
import os
import math
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import re
from collections import defaultdict
import json
from sympy import sympify, SympifyError
import utils.scorer
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from openai import OpenAI
from evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
# import matplotlib.pyplot as plt
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--use_wait_more", action="store_true")
    parser.add_argument("--use_s1", action="store_true")
    parser.add_argument('--task_folder', type=str, default='anonymous_run')
    parser.add_argument("--alpha", default=1.4, type=float)
    parser.add_argument("--use_weighted_majority_vote", action="store_true")
    parser.add_argument("--use_dynamic_reasoning", action="store_true")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--is_med', action="store_true")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    print("Args: ", args)

    return args



def get_reasoning_strategy(score):
    """
    根据问题的知识性/推理性评分，返回推理策略
    
    Args:
        score: 0-5的评分，0为纯知识型，5为纯推理型
    
    Returns:
        'parallel_samples': int, 'max_tokens': int
    """
    parallel_samples = max(3, round(8 - score))
    max_tokens = min(8192, round(2048 + score * 1229))
    
    return parallel_samples, max_tokens

def get_question_score(question, client):
    """
    调用外部大模型对问题进行评分
    
    Args:
        question: 问题文本
        client: OpenAI客户端
    
    Returns:
        int: 0-5的评分，0为纯知识型，5为纯推理型
    """
    scoring_prompt = f"""请对以下问题进行评分，评分标准如下：
- 0分：纯知识型问题，主要依赖事实记忆和知识检索
- 1分：知识为主，少量推理
- 2分：偏知识型，需要一定推理
- 3分：平衡型，知识和推理并重
- 4分：偏推理型，需要较多逻辑推理
- 5分：重推理型问题，需要强大的推理能力

问题：{question}

请只回答一个0-5之间的数字。"""

    try:
        completion = client.chat.completions.create(
            model="qwen-plus-latest", 
            messages=[
                {'role': 'user', 'content': scoring_prompt},
            ]
        )
        
        # 从回复中提取数字
        response = completion.choices[0].message.content.strip()
        # 尝试提取数字
        import re
        numbers = re.findall(r'[0-5]', response)
        if numbers:
            return int(numbers[0])
        else:
            return 2  # 默认为平衡型
    except Exception as e:
        print(f"评分失败: {e}")
        return 2  # 默认为平衡型


def generate_with_s1_budget_forcing(llm, input_prompts, max_tokens, temperature, top_p, seed, stop_words):
    """
    使用S1 budget forcing生成回答
    
    Args:
        llm: vLLM模型
        input_prompts: 输入提示列表
        max_tokens: 最大token数
        temperature: 温度参数
        top_p: top_p参数
        seed: 随机种子
        stop_words: 停止词列表
    
    Returns:
        list: 生成的回答列表
    """
    outputs = []
    
    # 第一阶段：生成直到遇到</think>或其他停止词
    gen_output = llm.generate(
        input_prompts, 
        SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=max_tokens, 
            seed=seed,
            n=1, 
            stop=stop_words+["</think>"],
            include_stop_str_in_output=True,
        )
    )
    
    # 处理第一阶段输出
    output_responses = []
    remaining_tokens = []
    prompts = input_prompts.copy()
    
    for index in range(len(input_prompts)):
        response = gen_output[index].outputs[0].text
        response = response.replace("</think>", "Wait")
        output_responses.append(response)
        prompts[index] += response
        remaining_tokens.append(max(1, max_tokens - len(gen_output[index].outputs[0].token_ids)))
    
    # 第二阶段：继续生成，仍然监听</think>标记
    sampling_params_list = [
        SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=remaining_token,
            seed=seed,
            n=1, 
            stop=stop_words+["</think>"],
            include_stop_str_in_output=True,
        ) for remaining_token in remaining_tokens
    ]
    gen_output = llm.generate(prompts, sampling_params_list)

    # 处理第二阶段输出
    for index in range(len(prompts)):
        response = gen_output[index].outputs[0].text
        response = response.replace("</think>", "Wait")
        output_responses[index] += response
        prompts[index] += response
        remaining_tokens[index] = max(1, remaining_tokens[index] - len(gen_output[index].outputs[0].token_ids))
    
    # 第三阶段：最终生成，移除</think>停止条件
    sampling_params_list = [
        SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=remaining_token,
            seed=seed,
            n=1, 
            stop=stop_words,
            include_stop_str_in_output=True,
        ) for remaining_token in remaining_tokens
    ]
    gen_output = llm.generate(prompts, sampling_params_list)
    
    # 处理第三阶段输出
    for index in range(len(prompts)):
        response = gen_output[index].outputs[0].text
        output_responses[index] += response

    # 检查是否需要继续生成（防止截断）
    for idx_prompt, full_output in enumerate(output_responses):
        n = len(llm.get_tokenizer().encode(full_output))
        if _need_continue(full_output, n, max_tokens):
            full_output += _continue(llm,
                                     input_prompts[idx_prompt] + full_output,
                                     max_tokens - n,
                                     stop_words)
        outputs.append(full_output)
    
    return outputs


def main_dynamic_reasoning(llm, tokenizer, data_name, args):
    """
    使用动态推理策略的主函数
    基于外部评分 -> 动态采样策略 -> S1 budget forcing -> 加权多数投票
    """
    # 准备数据
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 初始化OpenAI客户端用于评分
    client = OpenAI(
        api_key=" ${replace with your api key}",  
        base_url="${replace with your LLM URL}"
    )

    # 设置prompt模板
    query_prompt = (
        "{question}\n{option_str}\nConclude with your final answer enclosed in \\boxed{{}}." if not args.strict_prompt else
        "{question}\n{option_str}\nYour response must end with the correct option in the form \\boxed{{A}}."
    )

    # 准备所有prompt
    for example in examples:
        example['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in example['options'].items()])
        example["input_str"] = query_prompt.format_map(example)

    # 设置停止词
    stop_words = ["<|im_end|>", "<|im_start|>"]

    start_time = time.time()
    
    if args.use_vllm:
        final_outputs = []
        
        for i, example in enumerate(examples):
            print(f"Processing example {i+1}/{len(examples)}")
            
            # 步骤1: 获取问题评分
            question_text = example['question']
            score = get_question_score(question_text, client)
            print(f"Question score: {score}")
            
            # 步骤2: 根据评分获取动态策略
            parallel_samples, max_tokens = get_reasoning_strategy(score)
            print(f"Strategy: {parallel_samples} samples, {max_tokens} max_tokens")
            
            # 步骤3: 准备当前问题的输入
            current_input = example["input_str"]
            if args.apply_chat_template:
                current_input = tokenizer.apply_chat_template(
                    [{"role": "user", "content": current_input.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            
            # 步骤4: 生成多个样本，每个样本使用S1 budget forcing
            all_sample_outputs = []
            for sample_idx in range(parallel_samples):
                print(f"Generating sample {sample_idx+1}/{parallel_samples}")
                
                # 为每个样本使用不同的种子确保多样性
                sample_seed = args.seed + sample_idx * 1000 + i
                
                # 使用S1 budget forcing生成单个样本
                sample_outputs = generate_with_s1_budget_forcing(
                    llm=llm,
                    input_prompts=[current_input],
                    max_tokens=max_tokens,
                    temperature=max(0.7, args.temperature),  # 确保有一定随机性
                    top_p=args.top_p,
                    seed=sample_seed,
                    stop_words=stop_words
                )
                
                # 计算token长度
                sample_text = sample_outputs[0]
                token_length = len(llm.get_tokenizer().encode(sample_text))
                all_sample_outputs.append((sample_text, token_length))
            
            # 步骤5: 使用加权多数投票选择最终答案
            final_answer = weighted_majority_vote(all_sample_outputs, tokenizer)
            
            # 如果无法确定答案，使用第一个输出
            if final_answer is None:
                final_answer = all_sample_outputs[0][0] if all_sample_outputs else ""
            
            # 确保答案格式正确
            if final_answer and not final_answer.startswith("\\boxed{"):
                extracted = extract_answer(final_answer)
                if extracted:
                    final_answer = f"\\boxed{{{extracted}}}"
            
            final_outputs.append(final_answer)
            
    else:
        raise NotImplementedError("Only vLLM is supported in this simplified version")

    time_use = time.time() - start_time

    # 准备结果目录
    task_folder = os.path.join('./results', args.task_folder)
    os.makedirs(os.path.join(task_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(task_folder, 'result'), exist_ok=True)

    # 保存原始输出
    final_results = []
    for example, output in zip(examples, final_outputs):
        example["output"] = output
        final_results.append(example)

    # 生成任务名称
    task_name = "medical_dynamic_reasoning"
    if args.strict_prompt:
        task_name += "_strict-prompt"

    # 保存日志和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.join(task_folder, 'logs', f"{task_name}_{timestamp}.json")
    with open(log_save_path, 'w', encoding='utf-8') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 计算指标
    result_json = utils.scorer.get_results(log_save_path)
    result_json.update({
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "dynamic_reasoning": True
    })

    metrics_path = os.path.join(task_folder, 'result', f"{task_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)

    return result_json

def _need_continue(text, n_tokens, limit):
    return n_tokens >= limit - 5 and "\\boxed{" not in text

def _continue(llm, prompt_so_far, max_new, stop):
    max_new = max(max_new,2048)
    return llm.generate([prompt_so_far.rstrip() + "\n\\boxed{"],
                        SamplingParams(temperature=0, max_tokens=max_new, stop=stop))[0].outputs[0].text

def prepare_data(data_name, args):
    examples = load_data(data_name, args ,args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def extract_answer(text):
    """
    Extract answer from different model output formats
    
    Args:
        text: The model output text
        
    Returns:
        str: extracted answer or None if no answer found
    """
    
    # For HuatuoGPT-o1
    if '## Final Response\n\n' in text:
        text = text.split('## Final Response\n\n')[-1]
    # for our model
    elif '## Final Answer\n\n' in text:
        text = text.split('## Final Answer\n\n')[-1]
    # for deep seek distill qwen:
    elif '\n\nanswer:' in text.lower():
        text = text.lower().split('\n\nanswer:')[-1]
    if not text:
        return None
    
    # 只检查 \boxed{} 格式
    boxed_patterns = [
        r'\\boxed\{([A-K])\}',  # \boxed{A}
        r'\\boxed\{\s*([A-K])\s*\}',  # \boxed{ A }
        r'boxed\{([A-K])\}',  # boxed{A} (without backslash)
        r'\\boxed\{([A-K])\s*\}',  # \boxed{A }
        r'\\boxed\{\s*([A-K])\}',  # \boxed{ A}
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 取最后一个匹配的答案
            answer = matches[-1].upper()
            return answer
    
    return None

    

def weighted_majority_vote(answers_with_lengths, tokenizer=None):
    """
    使用加权多数投票选择最终答案
    公式: count / log(avg_length)
    
    Args:
        answers_with_lengths: list of tuples (answer_text, token_length)
        tokenizer: 用于计算token长度的tokenizer
    
    Returns:
        最终选择的答案
    """
    if not answers_with_lengths:
        return None
    
    # 提取所有答案选项
    extracted_answers = []
    for answer_text, token_length in answers_with_lengths:
        extracted_answer = extract_answer(answer_text)
        if extracted_answer:
            extracted_answers.append((extracted_answer, token_length))
    
    if not extracted_answers:
        return None
    
    # 统计每个答案的出现次数和token长度
    answer_stats = defaultdict(lambda: {'count': 0, 'total_length': 0})
    
    for answer, length in extracted_answers:
        answer_stats[answer]['count'] += 1
        answer_stats[answer]['total_length'] += length
    
    # 计算每个答案的权重
    answer_weights = {}
    for answer, stats in answer_stats.items():
        count = stats['count']
        avg_length = stats['total_length'] / count
        
        #避免log(0)的情况，如果avg_length <= 1，使用log(2)
        if avg_length <= 1:
            weight = count / math.log(2)
        else:
            weight = count / math.log(avg_length)
        #weight = count
        
        answer_weights[answer] = weight
    
    # 选择权重最大的答案
    if answer_weights:
        best_answer = max(answer_weights, key=answer_weights.get)
        print(f"Answer weights: {answer_weights}")
        print(f"Selected answer: {best_answer}")
        return best_answer
    
    return None

def main_weighted_majority_vote(llm, tokenizer, data_name, args):
    """
    使用加权多数投票的主函数
    """
    # 准备数据
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 设置prompt模板
    query_prompt = (
        "{question}\n{option_str}\nConclude with your final answer enclosed in \\boxed{{}}." if not args.strict_prompt else
        "{question}\n{option_str}\nYour response must end with the correct option in the form \\boxed{{A}}."
    )



    # 准备所有prompt
    for example in examples:
        example['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in example['options'].items()])
        example["input_str"] = query_prompt.format_map(example)

    input_prompts = [item["input_str"] for item in examples]

    # 应用chat模板
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    # 设置停止词
    stop_words = ["<|im_end|>", "<|im_start|>"]
    # if args.prompt_type == "cot":
    #     stop_words.append("\n\nQuestion:")
    # elif args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])

    start_time = time.time()
    
    if args.use_vllm:
        # 使用更高的温度和多次采样
        sampling_temperature = max(0.7, args.temperature)  # 确保有一定的随机性
        n_samples = max(args.n_sampling, 5)  # 至少采样5次
        
        print(f"Using temperature: {sampling_temperature}, n_samples: {n_samples}")
        
        gen_output = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=sampling_temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                seed=args.seed,
                n=n_samples,  # 多次采样
                # stop=stop_words
            )
        )
        
        # 处理多次采样的结果
        final_outputs = []
        for i, example in enumerate(examples):
            # 获取当前样本的所有采样结果
            sample_outputs = gen_output[i].outputs
            
            # 计算每个输出的token长度并收集答案
            answers_with_lengths = []
            # for output in sample_outputs:
            #     answer_text = output.text.strip()
            #     token_length = len(output.token_ids)
            #     answers_with_lengths.append((answer_text, token_length))
            # 加入防止截断
            for output in sample_outputs:
                answer_text = output.text.strip()
                token_length = len(output.token_ids)
                if _need_continue(answer_text, token_length, args.max_tokens_per_call):
                    answer_text += _continue(llm,
                                             example["input_str"] + answer_text,
                                             args.max_tokens_per_call - token_length,
                                             stop_words)
                    token_length = len(llm.get_tokenizer().encode(answer_text))
                answers_with_lengths.append((answer_text, token_length))
            # 使用加权多数投票选择最终答案
            final_answer = weighted_majority_vote(answers_with_lengths, tokenizer)
            
            # 如果无法确定答案，使用第一个输出
            if final_answer is None:
                final_answer = answers_with_lengths[0][0] if answers_with_lengths else ""
            boxed_answer = f"\\boxed{{{final_answer}}}"
            final_outputs.append(boxed_answer)
            # final_outputs.append(final_answer)
    else:
        raise NotImplementedError("Only vLLM is supported in this simplified version")

    time_use = time.time() - start_time

    # 准备结果目录
    task_folder = os.path.join('./results', args.task_folder)
    os.makedirs(os.path.join(task_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(task_folder, 'result'), exist_ok=True)

    # 保存原始输出
    final_results = []
    for example, output in zip(examples, final_outputs):
        example["output"] = output
        final_results.append(example)

    # 生成任务名称
    task_name = "medical_weighted_majority_vote"
    if args.strict_prompt:
        task_name += "_strict-prompt"

    # 保存日志和结果
    log_save_path = os.path.join(task_folder, 'logs', f"{task_name}.json")
    with open(log_save_path, 'w', encoding='utf-8') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 计算指标
    result_json = utils.scorer.get_results(log_save_path)
    result_json.update({
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "n_samples": n_samples,
        "sampling_temperature": sampling_temperature
    })

    metrics_path = os.path.join(task_folder, 'result', f"{task_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)

    return result_json


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        if args.use_wait_more:
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True,
                # enable_prefix_caching=True,
                # quantization='fp8',
                # kv_cache_dtype="fp8",
            )
        else:
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                trust_remote_code=True, 
                # kv_cache_dtype="fp8",
                # quantization='fp8',
            )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        if args.use_wait_more:
            if "1.5B" in args.model_name_or_path:
                if data_name == 'aime24':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 4500)
                elif data_name == 'amc23':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3000)
                elif data_name == 'minerva_math':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3200)
                elif data_name == 'math500':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2450)
                elif data_name == 'olympiadbench':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3410)
                else:
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 4500)
            elif "7B" in args.model_name_or_path or "8B" in args.model_name_or_path:
                if data_name == 'aime24':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 4600)
                elif data_name == 'amc23':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3250)
                elif data_name == 'minerva_math':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3080)
                elif data_name == 'math500':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3100)
                elif data_name == 'olympiadbench':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 3400)
                else:
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 4600)
            elif "32B" in args.model_name_or_path:
                if data_name == 'aime24':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2650)
                elif data_name == 'amc23':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2320)
                elif data_name == 'minerva_math':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 1800)
                elif data_name == 'math500':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2050)
                elif data_name == 'olympiadbench':
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2410)
                else:
                    args.threshold = int(args.max_tokens_per_call - args.alpha * 2650)
            results.append(main_wait_latest(llm, tokenizer, data_name, args))
        else:
            if args.use_s1:
                results.append(main_s1(llm, tokenizer, data_name, args))
            elif args.use_weighted_majority_vote:
                results.append(main_weighted_majority_vote(llm, tokenizer, data_name, args))
            elif args.use_dynamic_reasoning:
                results.append(main_dynamic_reasoning(llm, tokenizer, data_name, args))
            else:
                results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    # results.append(
    #     {
    #         "acc": sum([result["acc"] for result in results]) / len(results),
    #     }
    # )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def main(llm, tokenizer, data_name, args):
    # 准备数据
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 设置prompt模板
    query_prompt = (
        "Conclude with your final answer enclosed in \\boxed{{}}.\n"
        "{question}\n{option_str}\n" if not args.strict_prompt else
        "Your response must end with the correct option in the form \\boxed{{A}}.\n"
        "{question}\n{option_str}\n"
    )
    # 准备所有prompt
    for example in examples:
        example['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in example['options'].items()])
        example["input_str"] = query_prompt.format_map(example)

    input_prompts = [item["input_str"] for item in examples]

    # 应用chat
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    # 设置停止词
    stop_words = ["<|im_end|>", "<|im_start|>"]
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|end▁of▁sentence|>", " "]
    # if args.prompt_type == "cot":
    #     stop_words.append("\n\nQuestion:")
    # elif args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])

    start_time = time.time()
    if args.use_vllm:
        gen_output = llm.generate(
            input_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                seed=args.seed,
                n=1,
                # stop=stop_words
            )
        )
        #outputs = [output.outputs[0].text for output in gen_output]
        #加入防止截断
        outputs = []
        for out in gen_output:
            text = out.outputs[0].text
            n = len(out.outputs[0].token_ids)
            if _need_continue(text, n, args.max_tokens_per_call):
                text += _continue(llm,
                                  examples[len(outputs)]["input_str"] + text,
                                  args.max_tokens_per_call - n,
                                  stop_words)
            outputs.append(text)
    else:
        raise NotImplementedError("Only vLLM is supported in this simplified version")

    time_use = time.time() - start_time
    # print("time use:"+ time_use)

    # 准备结果目录
    task_folder = os.path.join('./results', args.task_folder)
    os.makedirs(os.path.join(task_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(task_folder, 'result'), exist_ok=True)

    # 保存原始输出
    final_results = []
    for example, output in zip(examples, outputs):
        example["output"] = output.strip()
        final_results.append(example)

    # 生成任务名称
    task_name = "medical"
    if args.strict_prompt:
        task_name += "_strict-prompt"

    # 保存日志和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.join(task_folder, 'logs', f"{task_name}_{timestamp}.json")
    # log_save_path = os.path.join(task_folder, 'logs', f"{task_name}.json")
    with open(log_save_path, 'w',encoding='utf-8') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 计算指标
    result_json = utils.scorer.get_results(log_save_path)
    result_json.update({
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    })

    metrics_path = os.path.join(task_folder, 'result', f"{task_name}_metrics.json")
    with open(metrics_path, 'w',encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)

    return result_json

def main_s1(llm, tokenizer, data_name, args):
    # 准备数据
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 设置prompt模板
    query_prompt = (
        "Conclude with your final answer enclosed in \\boxed{{}}.\n"
        "{question}\n{option_str}\n" if not args.strict_prompt else
        "Your response must end with the correct option in the form \\boxed{{A}}.\n"
        "{question}\n{option_str}\n"
    )
    # 准备所有prompt
    for example in examples:
        example['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in example['options'].items()])
        example["input_str"] = query_prompt.format_map(example)

    input_prompts = [item["input_str"] for item in examples]

    # 应用chat模板
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    # 设置停止词
    stop_words = ["<|im_end|>", "<|im_start|>"]
    # if args.prompt_type == "cot":
    #     stop_words.append("\n\nQuestion:")
    # elif args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])

    start_time = time.time()
    
    if args.use_vllm:
        outputs = []
        output_token_lengths = []
        
        # 第一阶段：生成直到遇到</think>或其他停止词
        gen_output = llm.generate(
            input_prompts, 
            SamplingParams(
                temperature=args.temperature, 
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call, 
                seed=args.seed,
                n=1, 
                stop=stop_words+["</think>"],
                include_stop_str_in_output=True,
            )
        )
        
        # 处理第一阶段输出
        output_responses = []
        remaining_tokens = []
        prompts = input_prompts.copy()
        
        for index in range(len(input_prompts)):
            response = gen_output[index].outputs[0].text
            response = response.replace("</think>", "Wait")
            output_responses.append(response)
            prompts[index] += response
            remaining_tokens.append(max(1, args.max_tokens_per_call - len(gen_output[index].outputs[0].token_ids)))
        
        # 第二阶段：继续生成，仍然监听</think>标记
        sampling_params_list = [
            SamplingParams(
                temperature=args.temperature, 
                top_p=args.top_p,
                max_tokens=remaining_token,
                seed=args.seed,
                n=1, 
                stop=stop_words+["</think>"],
                include_stop_str_in_output=True,
            ) for remaining_token in remaining_tokens
        ]
        gen_output = llm.generate(prompts, sampling_params_list)

        # 处理第二阶段输出
        for index in range(len(prompts)):
            response = gen_output[index].outputs[0].text
            response = response.replace("</think>", "Wait")
            output_responses[index] += response
            prompts[index] += response
            remaining_tokens[index] = max(1, remaining_tokens[index] - len(gen_output[index].outputs[0].token_ids))
        
        # 第三阶段：最终生成，移除</think>停止条件
        sampling_params_list = [
            SamplingParams(
                temperature=args.temperature, 
                top_p=args.top_p,
                max_tokens=remaining_token,
                seed=args.seed,
                n=1, 
                stop=stop_words,
                include_stop_str_in_output=True,
            ) for remaining_token in remaining_tokens
        ]
        gen_output = llm.generate(prompts, sampling_params_list)
        
        # 处理第三阶段输出
        for index in range(len(prompts)):
            response = gen_output[index].outputs[0].text
            output_responses[index] += response

        # 统计token长度
        # for idx_prompt, full_output in enumerate(output_responses):
        #     outputs.append(full_output)
        #     current_token_length = len(tokenizer.encode(full_output)) if tokenizer else len(full_output.split())
        #     output_token_lengths.append(current_token_length)
        #加入截断防止

        for idx_prompt, full_output in enumerate(output_responses):
            n = len(llm.get_tokenizer().encode(full_output))
            if _need_continue(full_output, n, args.max_tokens_per_call):
                full_output += _continue(llm,
                                         examples[idx_prompt]["input_str"] + full_output,
                                         args.max_tokens_per_call - n,
                                         stop_words)
            outputs.append(full_output)
        
        #average_token_length = sum(output_token_lengths) / len(output_token_lengths)
        #print(f"Average Token Length: {average_token_length:.2f}")
    else:
        raise NotImplementedError("Only vLLM is supported in this simplified version")

    time_use = time.time() - start_time

    # 准备结果目录
    task_folder = os.path.join('./results', args.task_folder)
    os.makedirs(os.path.join(task_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(task_folder, 'result'), exist_ok=True)

    # 保存原始输出
    final_results = []
    for example, output in zip(examples, outputs):
        example["output"] = output.strip()
        final_results.append(example)

    # 生成任务名称
    task_name = "medical_s1"
    if args.strict_prompt:
        task_name += "_strict-prompt"

    # 保存日志和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.join(task_folder, 'logs', f"{task_name}_{timestamp}.json")
    #log_save_path = os.path.join(task_folder, 'logs', f"{task_name}.json")
    with open(log_save_path, 'w', encoding='utf-8') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 计算指标
    result_json = utils.scorer.get_results(log_save_path)
    result_json.update({
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    })

    metrics_path = os.path.join(task_folder, 'result', f"{task_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)

    return result_json


def main_wait_latest(llm, tokenizer, data_name, args):
    # 准备数据
    examples, _, _ = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 设置prompt模板
    query_prompt = (
        "Conclude with your final answer enclosed in \\boxed{{}}.\n"
        "{question}\n{option_str}\n" if not args.strict_prompt else
        "Your response must end with the correct option in the form \\boxed{{A}}.\n"
        "{question}\n{option_str}\n"
    )

    # 准备所有prompt
    for example in examples:
        example['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in example['options'].items()])
        example["input_str"] = query_prompt.format_map(example)

    input_prompts = [item["input_str"] for item in examples]

    # 应用chat模板
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]

    # 设置停止词
    stop_words = ["<|im_end|>", "<|im_start|>"]
    # if args.prompt_type == "cot":
    #     stop_words.append("\n\nQuestion:")
    # elif args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
    #     stop_words.extend(["\n\n---", "```output"])

    start_time = time.time()
    
    if args.use_vllm:
        outputs = []
        output_token_lengths = []

        # NewlineWait类 - 动态等待机制
        class NewlineWait:
            def __init__(self, tokenizer, max_token_per_call=0, threshold=0):
                self.newline_ids = tokenizer(["\n\n", ",\n\n", ".\n\n", "]\n\n",
                                            ")\n\n", "],\n\n", "].\n\n", "].\n\n",
                                            ").\n\n", ".)\n\n"], add_special_tokens=False).input_ids
                self.newline_ids = list(chain.from_iterable(self.newline_ids))
                self.wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
                self.think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
                self.max_token_per_call = max_token_per_call
                self.threshold = threshold

            def __call__(self, token_ids, logits):
                if len(token_ids) < 2:
                    return logits
                
                remaining_tokens = self.max_token_per_call - len(token_ids)
                if remaining_tokens >= self.threshold and token_ids[-1] in self.newline_ids:
                    p_wait = (remaining_tokens - self.threshold) / (self.max_token_per_call - self.threshold)
                    if random.random() < p_wait:
                        logits.fill_(-float("inf"))
                        logits[self.wait_id] = 0.0
                return logits
        
        # 初始生成
        logits_processor = NewlineWait(llm.get_tokenizer(), max_token_per_call=args.max_tokens_per_call, threshold=args.threshold)
        gen_output = llm.generate(
            input_prompts, 
            SamplingParams(
                temperature=args.temperature, 
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call - args.threshold, 
                seed=args.seed,
                n=1, 
                stop=stop_words,
                include_stop_str_in_output=True,
                logits_processors=[logits_processor]
            )
        )

        # 处理初始输出并准备续写
        output_responses = []
        remaining_tokens_ = []
        prompts_ = []
        for index in range(len(input_prompts)):
            output_responses.append(gen_output[index].outputs[0].text)
            prompts_.append(input_prompts[index] + gen_output[index].outputs[0].text)
            remaining_tokens_.append(args.max_tokens_per_call - len(gen_output[index].outputs[0].token_ids))
        
        assert len(prompts_) == len(remaining_tokens_)
        stop_at_wait_words = ["Wait", ".Wait", "Wait, ", "Wait,"]
       
        active_traj = np.ones(len(prompts_))
        
        # 动态续写循环
        while 1:
            sampling_params_list = [
                SamplingParams(
                    temperature=args.temperature, 
                    top_p=args.top_p,
                    max_tokens=remaining_token_, 
                    seed=args.seed,
                    n=1, 
                    stop=stop_words + stop_at_wait_words,
                    include_stop_str_in_output=True,
                ) for remaining_token_ in remaining_tokens_
            ]
            
            input_prompts_while = [q for f, q in zip(active_traj, prompts_) if f]
            input_sampling_params_list = [q for f, q in zip(active_traj, sampling_params_list) if f]
            gen_output = llm.generate(input_prompts_while, input_sampling_params_list)
            
            i = 0
            for index in range(len(prompts_)):
                if active_traj[index] == 1:
                    response = gen_output[i].outputs[0].text
                    response = response.replace("\nWait", "</think>")
                    response = response.replace("Wait", "</think>")
                    
                    output_responses[index] += response
                    prompts_[index] += response
                    remaining_tokens_[index] = max(1, remaining_tokens_[index] - len(gen_output[i].outputs[0].token_ids))

                    if response.endswith(tuple(stop_words)) or remaining_tokens_[index] == 1 or response == "":
                        active_traj[index] = 0
                    i += 1
                
            print("ACTIVE TRAJ: ", sum(active_traj))
            if sum(active_traj) == 0:
                break

        # 统计token长度
        for idx_prompt, full_output in enumerate(output_responses):
            outputs.append(full_output)
            current_token_length = len(tokenizer.encode(full_output)) if tokenizer else len(full_output.split())
            output_token_lengths.append(current_token_length)
            
        average_token_length = sum(output_token_lengths) / len(output_token_lengths)
        print(f"Average Token Length: {average_token_length:.2f}")
    else:
        raise NotImplementedError("Only vLLM is supported in this simplified version")

    time_use = time.time() - start_time

    # 准备结果目录
    task_folder = os.path.join('./results', args.task_folder)
    os.makedirs(os.path.join(task_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(task_folder, 'result'), exist_ok=True)

    # 保存原始输出
    final_results = []
    for example, output in zip(examples, outputs):
        example["output"] = output.strip()
        final_results.append(example)

    # 生成任务名称
    task_name = "medical_wait_more"
    if args.strict_prompt:
        task_name += "_strict-prompt"

    # 保存日志和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = os.path.join(task_folder, 'logs', f"{task_name}_{timestamp}.json")
    #log_save_path = os.path.join(task_folder, 'logs', f"{task_name}.json")
    with open(log_save_path, 'w', encoding='utf-8') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # 计算指标
    result_json = utils.scorer.get_results(log_save_path)
    result_json.update({
        "time_use_in_second": time_use,
        "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    })

    metrics_path = os.path.join(task_folder, 'result', f"{task_name}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=4)

    return result_json

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)