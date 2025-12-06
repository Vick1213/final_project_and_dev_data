import json
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL = os.getenv("MODEL_NAME", "bens_model")

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                maxtoken: int = 128,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": maxtoken,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def calculator(text):
    text_lower = text.lower()
    
    if "calculator(" in text_lower:
        start = text_lower.find("calculator(") + len("calculator(")
        end = text.find(")", start)
        if end != -1:
            expr = text[start:end].strip()
        else:
            expr = ""
    else:
        expr = ""
        for char in text:
            if char in "0123456789+-*/.() ":
                expr += char
        expr = expr.strip()
    
    if not expr:
        return "Error: No expression found"
    
    try:
        expr = expr.replace("^", "**")
        result = eval(expr)
        return str(result)
    except Exception as e:
        return "Error: " + str(e)


def python_runner(text):
    code = ""
    
    if "```python" in text.lower():
        start_marker = "```python"
        start = text.lower().find(start_marker) + len(start_marker)
        end = text.find("```", start)
        if end != -1:
            code = text[start:end].strip()
    
    elif "python_runner:" in text.lower():
        start = text.lower().find("python_runner:") + len("python_runner:")
        code = text[start:].strip()
    
    elif "`" in text:
        start = text.find("`") + 1
        end = text.find("`", start)
        if end != -1:
            code = text[start:end].strip()
    
    if not code:
        return "Error: No code found"
    
    try:
        local_vars = {}
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted
        }
        exec(code, {"__builtins__": safe_builtins}, local_vars)
        if "result" in local_vars:
            return str(local_vars["result"])
        elif local_vars:
            return str(local_vars)
        else:
            return "Code ran successfully (no output)"
    except Exception as e:
        return "Error: " + str(e)


def chain_of_thought(text):
    problem = ""
    if "cot(" in text.lower():
        start = text.lower().find("cot(") + len("cot(")
        end = text.find(")", start)
        if end != -1:
            problem = text[start:end].strip()
    else:
        problem = text
    
    if not problem:
        return "Error: No problem found for CoT analysis"
    
    difficulty_prompt = f"""Rate the difficulty of this problem on a scale of 1-10.
1-3 = Easy (basic arithmetic, simple facts)
4-6 = Medium (multi-step reasoning, some complexity)
7-10 = Hard (complex math, logic puzzles, multi-step proofs)

Problem: {problem}

Reply with ONLY a number from 1-10."""
    
    difficulty_response = call_model_chat_completions(
        difficulty_prompt,
        system="You are a problem difficulty assessor. Reply with only a number 1-10.",
        temperature=0.0
    )
    
    difficulty = 5
    if difficulty_response['ok']:
        try:
            for char in difficulty_response['text']:
                if char.isdigit():
                    difficulty = int(char)
                    break
        except:
            difficulty = 5
    
    if difficulty <= 4:
        return f"[Easy problem, difficulty={difficulty}] Think simply and give the direct answer."
    
    cot_prompt = f"""Solve this problem step by step. Think carefully through each step.

Problem: {problem}

Let's think step by step:
1. First, identify what we're asked to find.
2. Break down the problem into smaller parts.
3. Solve each part carefully.
4. Combine the results for the final answer.

Show your reasoning, then give the final answer."""
    
    cot_response = call_model_chat_completions(
        cot_prompt,
        system="You are a careful problem solver. Think step by step before answering.",
        temperature=0.0,
        maxtoken=512
    )
    
    if cot_response['ok']:
        reasoning = cot_response['text'].strip()
        return f"[Hard problem, difficulty={difficulty}] Step-by-step reasoning:\n{reasoning}"
    else:
        return f"Error in CoT reasoning: {cot_response['error']}"


def python_code_returner(text):
    problem = text
    
    prompt = f"""Write Python code to solve this problem.
Return ONLY the function body code - no explanations, no markdown, no comments.
Use standard library imports if needed.
Match the expected function signature if one is provided in the problem.

Problem: {problem}

Output ONLY executable Python code:"""
    
    response = call_model_chat_completions(
        prompt,
        system="You are a code generator. Output ONLY Python code. No explanations. No markdown. Just raw executable code that solves the problem.",
        temperature=0.0,
        maxtoken=768
    )
    
    if response['ok']:
        code = response['text'].strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    return "Error generating code"


tools = [
    {
        "name": "calculator",
        "description": "Evaluates math expressions. Use: calculator(2 + 3 * 4)",
        "function": calculator
    },
    {
        "name": "python_runner",
        "description": "Runs Python code and returns the result. Use when you need to EXECUTE code: ```python your_code ```",
        "function": python_runner
    },
    {
        "name": "cot",
        "description": "Chain-of-Thought reasoning. Use for complex problems: cot(problem description)",
        "function": chain_of_thought
    },
    {
        "name": "python_code_returner",
        "description": "Generates Python code as the FINAL ANSWER. Use when asked to WRITE/IMPLEMENT a function or code. Use: python_code_returner(task description). Returns the code itself, not execution result.",
        "function": python_code_returner
    }
]


def agent_loop(question: str, tools: list, history: list = None, max_steps: int = 6):
    if history is None:
        history = []
    
    tool_descriptions = "\n".join([f"- {t['name']}: {t.get('description', 'No description')}" for t in tools])
    system_prompt = f"""You are a helpful assistant with access to these tools:
{tool_descriptions}

When you want to use a tool, say: USE TOOL: tool_name(arguments)
When you have the final answer, say: FINAL ANSWER: your answer here

IMPORTANT RULES:
1. For True/False questions, answer with EXACTLY "True" or "False" (nothing else)
2. For numeric questions, answer with just the number
3. For coding questions, use python_code_returner tool
4. Always end with FINAL ANSWER: followed by just the answer, nothing else
"""
    
    for step in range(max_steps):
        if history:
            history_text = "Previous steps:\n" + "\n".join(history)
            prompt = f"Question: {question}\n\n{history_text}\n\nWhat do you want to do next? Remember to say FINAL ANSWER: when done."
        else:
            prompt = f"Question: {question}\n\nWhat do you want to do next?"
        
        response = call_model_chat_completions(prompt, system=system_prompt, maxtoken=256)
        if not response['ok']:
            print(f"Model call failed: {response['error']}")
            return None
        
        model_output = response['text'].strip()
        print(f"Step {step + 1}: {model_output}")
        
        lower_output = model_output.lower()
        if "final answer:" in lower_output:
            idx = lower_output.find("final answer:")
            final_answer = model_output[idx + len("final answer:"):].strip()
            
            if "\\boxed{" in final_answer:
                start = final_answer.find("\\boxed{") + 7
                end = final_answer.rfind("}")
                if end > start:
                    final_answer = final_answer[start:end]
            
            lower_answer = final_answer.lower()
            words = lower_answer.split()
            first_word = words[0] if words else ""
            if "false" in lower_answer and "true" not in lower_answer:
                final_answer = "False"
            elif "true" in lower_answer and "false" not in lower_answer:
                final_answer = "True"
            elif first_word in ["no", "no,", "no."]:
                final_answer = "False"
            elif first_word in ["yes", "yes,", "yes."]:
                final_answer = "True"
            
            print(f">>> Final answer: {final_answer}")
            return final_answer
        
        tool_called = False
        for tool in tools:
            if tool['name'].lower() in lower_output:
                tool_result = tool['function'](model_output)
                
                if tool['name'] == "python_code_returner":
                    print(f"    Generated code:\n{tool_result[:200]}...")
                    return tool_result
                
                history.append(f"Step {step + 1}: Called {tool['name']}, got result: {tool_result}")
                print(f"    Tool result: {tool_result}")
                tool_called = True
                break
        
        if not tool_called:
            history.append(f"Step {step + 1}: {model_output[:200]}")
    
    print("Max steps reached without final answer")
    if history:
        last = history[-1]
        if "result:" in last:
            return last.split("result:")[-1].strip()
    return None


def run_chunk(chunk_num, questions, total_chunks=8):
    total = len(questions)
    chunk_size = total // total_chunks
    
    if chunk_num < total_chunks - 1:
        start = chunk_num * chunk_size
        end = start + chunk_size
    else:
        start = chunk_num * chunk_size
        end = total
    
    print(f"[Chunk {chunk_num + 1}] Processing questions {start + 1} to {end}")
    
    answers = []
    for idx, q in enumerate(questions[start:end], start=start + 1):
        ans = agent_loop(q["input"], tools=tools)
        if ans is None:
            ans = ""
        elif not isinstance(ans, str):
            ans = str(ans)
        ans = ans.strip()[:4999]
        answers.append({"output": ans})
        
        if (idx - start) % 10 == 0:
            print(f"[Chunk {chunk_num + 1}] Completed {idx - start}/{end - start}")
    
    output_file = f"answers_chunk{chunk_num + 1}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    print(f"[Chunk {chunk_num + 1}] DONE - Saved {len(answers)} answers to {output_file}")
    return chunk_num + 1, len(answers)


def merge_chunks(num_chunks=8):
    all_answers = []
    for i in range(1, num_chunks + 1):
        filename = f"answers_chunk{i}.json"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                all_answers.extend(chunk)
                print(f"Loaded {len(chunk)} answers from {filename}")
        except FileNotFoundError:
            print(f"WARNING: {filename} not found!")
    
    with open("cse_476_final_project_answers.json", "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal answers merged: {len(all_answers)}")
    print("Saved to cse_476_final_project_answers.json")


def main():
    input_file = "cse_476_final_project_test_data.json"
    
    print(f"Loading questions from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    total = len(questions)
    num_chunks = 8
    print(f"Total questions: {total}")
    print(f"Running {num_chunks} chunks in parallel...\n")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = {executor.submit(run_chunk, i, questions, num_chunks): i for i in range(num_chunks)}
        
        for future in as_completed(futures):
            chunk_num, count = future.result()
            print(f">>> Chunk {chunk_num} finished with {count} answers")
    
    elapsed = time.time() - start_time
    print(f"\nAll chunks completed in {elapsed/60:.1f} minutes")
    
    print("\nMerging all chunks...")
    merge_chunks(num_chunks)
    print("\nDONE!")


if __name__ == "__main__":
    main()