import math
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import torch
import json
import os
from tqdm import tqdm
import gc

class CxmiPruning:
    def __init__(self,model_name_or_path, tokenizer_name_or_path, prefix):
        CACHE_DIR = "/cs/student/projects1/ml/2024/yihanli/filco/hf_CACHE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16).to(self.device).eval()
        self.prefix = prefix
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=CACHE_DIR)

    def get_output_probs(self, input_text, output_text):
        """Compute the output probabilities of the output text given the input text."""
        input_dict = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)  # <1, in-len>
        output_dict = self.tokenizer(output_text, return_tensors="pt", truncation=True, max_length=512)  # <1, out-len>

        input_dict["labels"] = output_dict["input_ids"]
        input_dict = {k: v.to(self.device) for k, v in input_dict.items()}

        logits = self.model(**input_dict).logits  # <1, out-len, vocab-size>
        probs = logits.softmax(dim=-1).squeeze(0)  # <out-len, vocab-size>
        label_ids = input_dict["labels"].squeeze(0).unsqueeze(-1)  # <out-len, 1>
        probs = probs.gather(1, label_ids).squeeze(-1)  # <out-len>
        return probs.tolist()

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + math.exp(-x))

    def sent_wise_diff(self, base_scores, ctx_scores):
        """Compute the sentence-wise difference between context over the raw vector."""
        assert len(base_scores) == len(ctx_scores), "The two lists must have the same length."

        return -np.log(np.prod(base_scores) / np.prod(ctx_scores))

    def calc_cxmi_score(self, answer, base_input, ctx_input, apply_sigmoid=False):
        """Compute the CXMI score."""
        base_probs = self.get_output_probs(base_input, answer)
        ctx_probs = self.get_output_probs(ctx_input, answer)
        diff = self.sent_wise_diff(base_scores=base_probs, ctx_scores=ctx_probs)
        if apply_sigmoid:
            diff = self.sigmoid(diff)

        return diff

    def get_input_text(self,
        question,
        question_prefix="question",
        context=None,
        context_prefix="context",
    ):
        """Construct the input text."""
        q_text = f"{question_prefix}: {question}"
        if context is None:
            return q_text
        ctx_text = f"{context_prefix}: {context}"
        return "\n".join([ctx_text, q_text])


    def get_example_inputs(self,
        question,
        context,
        answers,
        question_prefix: str = "question",
        context_prefix: str = "context",
    ):
        """Get example inputs for the generation model."""
        base_input = self.get_input_text(
            question,
            context=None,
            question_prefix=question_prefix,
            context_prefix=context_prefix,
        )
        ctx_input = self.get_input_text(
            question,
            context=context,
            question_prefix=question_prefix,
            context_prefix=context_prefix,
        )
        return {
            "base_input": base_input,
            "ctx_input": ctx_input,
            "answers": answers,
        }

    def calc_text_scores(self, text, question, answers):
        """Calculate scores for a context text."""
        scores_dict = {}
        scores_dict["cxmi"] = self.calc_cxmi(text, question, answers)
        scores_dict["text"] = text
        return scores_dict

    def calc_cxmi(self, text, question, answers):
        """Calculate CXMI score for a context text."""
        proc_inputs = self.get_example_inputs(
            question=self.prefix + question,
            context=text,
            answers=answers,
        )
        cxmi_score = self.calc_cxmi_score(
            answer=proc_inputs["answers"][0],
            base_input=proc_inputs["base_input"],
            ctx_input=proc_inputs["ctx_input"],
            apply_sigmoid=True,
        )
        return cxmi_score

    def load_json_file(self, file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return None


    def prune(self, input_retrieved_path, answer_path, output_path, passage_level=False):
        retrieved_data = self.load_json_file(input_retrieved_path)
        answers_data = self.load_json_file(answer_path)
        print(f"answers_data length: {len(answers_data)}")

        pruned_ctxs = {}

        for idx, answer_dict in tqdm(enumerate(answers_data), total=len(answers_data), desc="CXMI Pruning"):
            content = retrieved_data[idx]
            query = content["query"]

            answers = [answer_dict["answer"]]
            all_sent_dicts = []

            for doc in content["retrieved"]:
                doc_content = doc.get("text") or doc.get("abstract", "")
                if doc_content == "":
                    continue

                if passage_level:
                    try:
                        score_dict = self.calc_text_scores(doc_content, query, answers)
                        all_sent_dicts.append(score_dict)
                    except Exception as e:
                        print(f"Error scoring sentence: {e}")
                else:
                    # Split document into sentences
                    sentences = sent_tokenize(doc_content)
                    for s in sentences:
                        try:
                            score_dict = self.calc_text_scores(s, query, answers)
                            all_sent_dicts.append(score_dict)
                        except Exception as e:
                            print(f"Error scoring sentence: {e}")

            if not all_sent_dicts:
                best_sentence = ""
            else:
                best_entry = max(all_sent_dicts, key=lambda x: x["cxmi"])
                best_sentence = best_entry["text"]

            pruned_ctxs[idx] = {
                "query": query,
                "retrieved_docs": [best_sentence]
            }

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as out_f:
            json.dump(pruned_ctxs, out_f, indent=2)
        print(f"File saved to {output_path}")

    def medmcqa_prune(self, input_retrieved_path, answer_path, output_path, passage_level=False):
        # Load data
        retrieved_data = self.load_json_file(input_retrieved_path)
        answers_data = self.load_json_file(answer_path)

        pruned_ctxs = {}
        seen_queries = set()

        for idx, answer_dict in tqdm(enumerate(answers_data), total=len(answers_data), desc="CXMI Pruning"):
            content = retrieved_data[idx]
            query = content["query"]

            # if query in seen_queries:
            #     continue
            # else:
            #     seen_queries.add(query)

            # answer_dict = answers_data[idx]

            # if query != answer_dict["question"]:
            #     raise ValueError("Query mismatch between retrieved and answer data")

            answer = answer_dict["cop"]
            answer_letter = chr(ord('A') + int(answer) - 1)
            answer_map = {
                "A": answer_dict["opa"],
                "B": answer_dict["opb"],
                "C": answer_dict["opc"],
                "D": answer_dict["opd"],
            }
            if answer_map["A"] == "":
                raise ValueError(f"Empty option A for idx {idx}")
            if answer_map["B"] == "":
                raise ValueError(f"Empty option B for idx {idx}")
            if answer_map["C"] == "":
                raise ValueError(f"Empty option C for idx {idx}")
            if answer_map["D"] == "":
                raise ValueError(f"Empty option D for idx {idx}")
            answers = [answer_map[answer_letter]]
            all_sent_dicts = []

            for doc in content["retrieved"]:
                doc_content = doc.get("text") or doc.get("abstract", "")
                if doc_content == "":
                    continue

                if passage_level:
                    try:
                        score_dict = self.calc_text_scores(doc_content, query, answers)
                        all_sent_dicts.append(score_dict)
                    except Exception as e:
                        print(f"Error scoring sentence: {e}")
                else:
                    # Split document into sentences
                    sentences = sent_tokenize(doc_content)
                    for s in sentences:
                        try:
                            score_dict = self.calc_text_scores(s, query, answers)
                            all_sent_dicts.append(score_dict)
                        except Exception as e:
                            print(f"Error scoring sentence: {e}")

            if not all_sent_dicts:
                best_sentence = ""
            else:
                best_entry = max(all_sent_dicts, key=lambda x: x["cxmi"])
                best_sentence = best_entry["text"]

            pruned_ctxs[idx] = {
                "query": query,
                "retrieved_docs": [best_sentence]
            }

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as out_f:
            json.dump(pruned_ctxs, out_f, indent=2)
        print(f"File saved to {output_path}")



if __name__=="__main__":
    ANSWER_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json"

    PREFIX = "Given the ['context', 'question'], predict the answer to the question:"

    sent_list = ["sent", "passage"]
    flant5_list = ["large", "base"]
    INPUT_PATH = f"/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_mmlu_cot_test_k_10.json"

    for sent_pas in sent_list:
        for flant5 in flant5_list:
            print(f"Processing {sent_pas} level with {flant5} model...")
            OUTPUT_PATH = f"/cs/student/projects1/ml/2024/yihanli/filco/output/mmlu_cot/cxmi_{sent_pas}_pruning_{flant5}_k_10.json"
            model_name = f"google/flan-t5-{flant5}"
            flag = sent_pas == "passage"
            print(f"passage level: {flag}")
            print("Training Starts....")
            filco = CxmiPruning(model_name_or_path=model_name, tokenizer_name_or_path=model_name, prefix=PREFIX)
            print("Model Initialized")
            filco.medmcqa_prune(input_retrieved_path=INPUT_PATH, answer_path=ANSWER_PATH, output_path=OUTPUT_PATH, passage_level=flag)

            del filco.model
            del filco.tokenizer
            del filco
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
